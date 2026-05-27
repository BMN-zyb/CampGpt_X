# train_sft.py

"""
SFT (Supervised Fine-Tuning) 全量微调
加载预训练权重 → 在对话数据上训练 → 保存SFT权重
"""

import os
import json
import math
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

os.environ["TIKTOKEN_CACHE_DIR"] = "./tiktoken_cache"
import tiktoken

from config import GPTConfig, get_model_config
from model import GPT


# =============================================================================
# ========================= SFT 配置 ==========================================
# =============================================================================

@dataclass
class SFTConfig:
    # 路径
    pretrained_path: str = "log/model_00999_CampGPT_X.pt"
    sft_data_path: str = "sft_dataset.json"
    output_dir: str = "sft_output"
    
    # 训练超参
    num_epochs: int = 100
    batch_size: int = 8
    max_seq_len: int = 512          # SFT 通常不需要很长
    learning_rate: float = 2e-5     # 比预训练小 10-30 倍
    min_lr: float = 2e-6
    warmup_steps: int = 10
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    grad_accum_steps: int = 1       # 有效batch = batch_size * grad_accum = 16
    
    # 对话格式
    system_prompt: str = "You are a helpful university assistant that answers questions about student policies and regulations."
    
    # 特殊 token
    # 用 GPT-2 tokenizer 中不常用的 token 作为分隔符
    user_token: str = "<|user|>"
    assistant_token: str = "<|assistant|>"
    end_token: str = "<|end|>"
    
    # 日志
    log_every: int = 10
    save_every_epoch: bool = False
    eval_ratio: float = 0.05        # 5% 数据做验证


# =============================================================================
# ========================= 对话模板 & Tokenizer ================================
# =============================================================================

class ChatTokenizer:
    """
    对话格式处理器
    格式: <|user|>question<|end|><|assistant|>answer<|end|>
    
    只对 assistant 的回答部分计算 loss (labels masking)
    """
    
    def __init__(self, config: SFTConfig):
        self.enc = tiktoken.get_encoding("gpt2")
        self.config = config
        self.vocab_size = self.enc.n_vocab  # 50257
        
        # 用 vocab 中已有但罕用的 token 做特殊标记
        # GPT-2 vocab 最后几个 token 很少出现
        self.user_token_id = 50256      # <|endoftext|> 复用不了，用其他的
        self.assistant_token_id = 50255
        self.end_token_id = 50254
        
        # 实际上 GPT-2 vocab 只到 50256，所以我们用特定字符串编码
        # 更简单的方案：用特定文本模式
        self.user_prefix = "\n\n### User:\n"
        self.assistant_prefix = "\n\n### Assistant:\n"
        self.turn_end = "\n\n"
        
        self.user_prefix_ids = self.enc.encode(self.user_prefix)
        self.assistant_prefix_ids = self.enc.encode(self.assistant_prefix)
        self.turn_end_ids = self.enc.encode(self.turn_end)
    
    def encode_conversation(self, messages: List[Dict], max_len: int) -> Dict:
        """
        将对话编码为 input_ids + labels
        只在 assistant 回复部分计算 loss
        
        Returns:
            input_ids: [token_ids]
            labels: [token_ids], user部分为 -100 (忽略)
        """
        input_ids = []
        labels = []
        
        # 可选：添加 system prompt
        if self.config.system_prompt:
            sys_text = f"### System:\n{self.config.system_prompt}\n\n"
            sys_ids = self.enc.encode(sys_text)
            input_ids.extend(sys_ids)
            labels.extend([-100] * len(sys_ids))  # system 不计算 loss
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "user":
                prefix_ids = self.user_prefix_ids
                content_ids = self.enc.encode(content)
                end_ids = self.turn_end_ids
                
                turn_ids = prefix_ids + content_ids + end_ids
                input_ids.extend(turn_ids)
                labels.extend([-100] * len(turn_ids))  # user 部分不计算 loss
                
            elif role == "assistant":
                prefix_ids = self.assistant_prefix_ids
                content_ids = self.enc.encode(content)
                end_ids = self.turn_end_ids
                
                turn_ids = prefix_ids + content_ids + end_ids
                input_ids.extend(turn_ids)
                # prefix 不计算 loss，content + end 计算 loss
                turn_labels = ([-100] * len(prefix_ids)) + content_ids + end_ids
                labels.extend(turn_labels)
        
        # 截断
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            labels = labels[:max_len]
        
        return {
            "input_ids": input_ids,
            "labels": labels,
        }
    
    def decode(self, token_ids: List[int]) -> str:
        """解码 token ids 为文本"""
        # 过滤特殊 token
        filtered = [t for t in token_ids if t < self.enc.n_vocab]
        return self.enc.decode(filtered)


# =============================================================================
# ========================= SFT 数据集 =========================================
# =============================================================================

class SFTDataset(Dataset):
    """SFT 对话数据集"""
    
    def __init__(self, data: List[Dict], tokenizer: ChatTokenizer, max_len: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item["messages"]
        encoded = self.tokenizer.encode_conversation(messages, self.max_len)
        return encoded


def sft_collate_fn(batch, max_len, pad_id=0):
    """
    动态 padding + 生成 attention mask
    """
    batch_input_ids = []
    batch_labels = []
    
    max_batch_len = min(max(len(b["input_ids"]) for b in batch), max_len)
    
    for b in batch:
        ids = b["input_ids"][:max_batch_len]
        labs = b["labels"][:max_batch_len]
        
        pad_len = max_batch_len - len(ids)
        ids = ids + [pad_id] * pad_len
        labs = labs + [-100] * pad_len  # padding 位置不计算 loss
        
        batch_input_ids.append(ids)
        batch_labels.append(labs)
    
    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
    }


# =============================================================================
# ========================= SFT 训练器 =========================================
# =============================================================================

class SFTTrainer:
    """SFT 全量微调训练器"""
    
    def __init__(self, sft_config: SFTConfig):
        self.sft_config = sft_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_type = "cuda" if self.device.startswith("cuda") else "cpu"
        
        # 初始化
        self._load_model()
        self._load_data()
        self._setup_optimizer()
        
    def _load_model(self):
        """加载预训练权重"""
        print(f"[SFT] Loading pretrained model from {self.sft_config.pretrained_path}")
        
        checkpoint = torch.load(self.sft_config.pretrained_path, map_location="cpu")
        model_config = checkpoint["config"]
        
        # 确保配置一致
        self.model_config = model_config
        self.model = GPT(model_config)
        
        # 加载权重
        state_dict = checkpoint["model"]
        # 处理可能的 key 不匹配（DDP/FSDP 保存的可能有 module. 前缀）
        cleaned = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "").replace("_orig_mod.", "")
            cleaned[k] = v
        
        self.model.load_state_dict(cleaned, strict=True)
        self.model.to(self.device)
        
        # 关闭 gradient checkpointing（SFT 数据量小，不需要）
        self.model.set_gradient_checkpointing(False)
        
        pretrain_step = checkpoint.get("step", "unknown")
        pretrain_loss = checkpoint.get("val_loss", "unknown")
        print(f"[SFT] Loaded pretrained model (step={pretrain_step}, val_loss={pretrain_loss})")
        print(f"[SFT] Model params: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _load_data(self):
        """加载并划分 SFT 数据"""
        print(f"[SFT] Loading data from {self.sft_config.sft_data_path}")
        
        with open(self.sft_config.sft_data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        # 打乱并划分
        random.seed(42)
        random.shuffle(raw_data)
        
        val_size = max(1, int(len(raw_data) * self.sft_config.eval_ratio))
        val_data = raw_data[:val_size]
        train_data = raw_data[val_size:]
        
        self.tokenizer = ChatTokenizer(self.sft_config)
        
        self.train_dataset = SFTDataset(train_data, self.tokenizer, self.sft_config.max_seq_len)
        self.val_dataset = SFTDataset(val_data, self.tokenizer, self.sft_config.max_seq_len)
        
        from functools import partial
        collate = partial(sft_collate_fn, max_len=self.sft_config.max_seq_len)
        
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.sft_config.batch_size,
            shuffle=True, 
            collate_fn=collate,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.sft_config.batch_size,
            shuffle=False,
            collate_fn=collate,
        )
        
        print(f"[SFT] Train: {len(train_data)} samples, Val: {len(val_data)} samples")
        
        # 打印一个样本
        sample = self.train_dataset[0]
        print(f"[SFT] Sample input length: {len(sample['input_ids'])} tokens")
        print(f"[SFT] Sample text preview: {self.tokenizer.decode(sample['input_ids'][:100])}...")
    
    def _setup_optimizer(self):
        """配置优化器"""
        self.optimizer = self.model.configure_optimizers(
            weight_decay=self.sft_config.weight_decay,
            learning_rate=self.sft_config.learning_rate,
            device_type=self.device_type,
        )
        
        # 计算总步数
        steps_per_epoch = len(self.train_loader) // self.sft_config.grad_accum_steps
        self.total_steps = steps_per_epoch * self.sft_config.num_epochs
        
        print(f"[SFT] Steps per epoch: {steps_per_epoch}, Total steps: {self.total_steps}")
    
    def _get_lr(self, step):
        """Cosine 学习率调度"""
        cfg = self.sft_config
        if step < cfg.warmup_steps:
            return cfg.learning_rate * (step + 1) / cfg.warmup_steps
        if step >= self.total_steps:
            return cfg.min_lr
        decay_ratio = (step - cfg.warmup_steps) / (self.total_steps - cfg.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)
    
    def _compute_loss(self, input_ids, labels):
        """计算带 label masking 的 loss"""
        logits, _ = self.model(input_ids)
        
        # Shift: logits[:-1] 预测 labels[1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,  # 忽略 user 部分和 padding
        )
        return loss
    
    @torch.no_grad()
    def evaluate(self):
        """验证集评估"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                loss = self._compute_loss(input_ids, labels)
            
            # 统计有效 token 数
            valid_tokens = (labels[:, 1:] != -100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
        
        avg_loss = total_loss / max(total_tokens, 1)
        self.model.train()
        return avg_loss
    
    def train(self):
        """主训练循环"""
        cfg = self.sft_config
        os.makedirs(cfg.output_dir, exist_ok=True)
        
        log_file = os.path.join(cfg.output_dir, "sft_log.txt")
        
        print(f"\n{'='*60}")
        print(f"  SFT Training Start")
        print(f"  Epochs: {cfg.num_epochs}")
        print(f"  Effective batch: {cfg.batch_size * cfg.grad_accum_steps}")
        print(f"  Learning rate: {cfg.learning_rate}")
        print(f"  Total steps: {self.total_steps}")
        print(f"{'='*60}\n")
        
        self.model.train()
        global_step = 0
        best_val_loss = float("inf")
        
        for epoch in range(cfg.num_epochs):
            epoch_loss = 0
            epoch_tokens = 0
            self.optimizer.zero_grad()
            
            for micro_step, batch in enumerate(self.train_loader):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    loss = self._compute_loss(input_ids, labels)
                
                loss_scaled = loss / cfg.grad_accum_steps
                loss_scaled.backward()
                
                valid_tokens = (labels[:, 1:] != -100).sum().item()
                epoch_loss += loss.item() * valid_tokens
                epoch_tokens += valid_tokens
                
                # 梯度累积步
                if (micro_step + 1) % cfg.grad_accum_steps == 0:
                    norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), cfg.grad_clip
                    )
                    
                    lr = self._get_lr(global_step)
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = lr
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    
                    if global_step % cfg.log_every == 0:
                        avg = epoch_loss / max(epoch_tokens, 1)
                        print(f"  epoch {epoch+1} step {global_step:4d} | "
                              f"loss {loss.item():.4f} | avg {avg:.4f} | "
                              f"lr {lr:.2e} | norm {norm:.4f}")
                        
                        with open(log_file, "a") as f:
                            f.write(f"{global_step} train {loss.item():.4f}\n")
            
            # Epoch 结束：验证
            val_loss = self.evaluate()
            avg_train = epoch_loss / max(epoch_tokens, 1)
            print(f"\n[Epoch {epoch+1}/{cfg.num_epochs}] "
                  f"train_loss={avg_train:.4f} val_loss={val_loss:.4f}")
            
            with open(log_file, "a") as f:
                f.write(f"epoch_{epoch+1} val {val_loss:.4f}\n")
            
            # # 保存
            # if cfg.save_every_epoch or val_loss < best_val_loss:
            #     if val_loss < best_val_loss:
            #         best_val_loss = val_loss
                
            #     ckpt_path = os.path.join(cfg.output_dir, f"sft_epoch_{epoch+1}.pt")
            #     torch.save({
            #         "model": self.model.state_dict(),
            #         "config": self.model_config,
            #         "sft_config": cfg,
            #         "epoch": epoch + 1,
            #         "global_step": global_step,
            #         "val_loss": val_loss,
            #         "chat_template": {
            #             "user_prefix": self.tokenizer.user_prefix,
            #             "assistant_prefix": self.tokenizer.assistant_prefix,
            #             "turn_end": self.tokenizer.turn_end,
            #             "system_prompt": cfg.system_prompt,
            #         },
            #     }, ckpt_path)
            #     print(f"  Saved: {ckpt_path}")
            
            # 保存 best
            if val_loss <= best_val_loss:
                best_path = os.path.join(cfg.output_dir, "sft_best.pt")
                torch.save({
                    "model": self.model.state_dict(),
                    "config": self.model_config,
                    "sft_config": cfg,
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "val_loss": val_loss,
                    "chat_template": {
                        "user_prefix": self.tokenizer.user_prefix,
                        "assistant_prefix": self.tokenizer.assistant_prefix,
                        "turn_end": self.tokenizer.turn_end,
                        "system_prompt": cfg.system_prompt,
                    },
                }, best_path)
                print(f"  Saved best: {best_path} (val_loss={val_loss:.4f})")
        
        print(f"\n[SFT] Training complete! Best val_loss: {best_val_loss:.4f}")
        
        # 快速生成测试
        self._test_generation()
    
    def _test_generation(self):
        """训练后快速测试生成"""
        print(f"\n{'='*40} Generation Test {'='*40}")
        
        test_questions = [
            "What are the requirements for applying for a scholarship?",
            "How do I transfer to a different major?",
            "What happens if I fail a course?",
        ]
        
        self.model.eval()
        enc = tiktoken.get_encoding("gpt2")
        
        for q in test_questions:
            prompt_text = ""
            if self.sft_config.system_prompt:
                prompt_text += f"### System:\n{self.sft_config.system_prompt}\n\n"
            prompt_text += f"### User:\n{q}\n\n### Assistant:\n"
            
            prompt_ids = enc.encode(prompt_text)
            prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    generated = self.model.generate(
                        prompt_t, max_new_tokens=150,
                        temperature=0.7, top_k=50, top_p=0.9
                    )
            
            output = enc.decode(generated[0].tolist())
            # 截取 assistant 回复部分
            if "### Assistant:" in output:
                answer = output.split("### Assistant:")[-1].strip()
                # 截断到第一个 ### 或换行结束
                if "###" in answer:
                    answer = answer[:answer.index("###")].strip()
            else:
                answer = output[len(prompt_text):]
            
            print(f"\nQ: {q}")
            print(f"A: {answer[:300]}")
        
        print(f"{'='*80}\n")


# =============================================================================
# ========================= 入口 ===============================================
# =============================================================================

if __name__ == "__main__":
    config = SFTConfig()
    
    # 可以命令行覆盖
    import sys
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, val = arg.split("=", 1)
            key = key.lstrip("-")
            if hasattr(config, key):
                field_type = type(getattr(config, key))
                setattr(config, key, field_type(val))
                print(f"  Override: {key} = {val}")
    
    trainer = SFTTrainer(config)
    trainer.train()