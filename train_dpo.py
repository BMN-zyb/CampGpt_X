# train_dpo.py

"""
DPO (Direct Preference Optimization) 训练
加载 SFT 权重 → 偏好优化 → 保存最终模型

DPO Loss:
L = -E[log σ(β · (log π(chosen) / π_ref(chosen) - log π(rejected) / π_ref(rejected)))]

参考: https://arxiv.org/abs/2305.18290
"""

import os
import json
import math
import time
import random
import copy
from dataclasses import dataclass
from typing import List, Dict
from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

os.environ["TIKTOKEN_CACHE_DIR"] = "./tiktoken_cache"
import tiktoken

from config import GPTConfig, get_model_config
from model import GPT
from train_sft import SFTConfig


# =============================================================================
# ========================= DPO 配置 ==========================================
# =============================================================================

@dataclass
class DPOConfig:
    # 路径
    sft_model_path: str = "sft_output/sft_best.pt"
    dpo_data_path: str = "dpo_dataset.json"
    output_dir: str = "dpo_output"
    
    # DPO 超参
    beta: float = 0.9               # KL 惩罚系数, 越大越保守
    num_epochs: int = 20
    batch_size: int = 8              # DPO 需要同时算 chosen+rejected, 显存翻倍
    max_seq_len: int = 512
    learning_rate: float = 5e-6      # 比 SFT 更小
    min_lr: float = 5e-7
    warmup_steps: int = 5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    grad_accum_steps: int = 1        # 有效 batch = 2 * 8 = 16
    
    # 对话格式（需要和 SFT 一致）
    system_prompt: str = "You are a helpful university assistant that answers questions about student policies and regulations."
    
    # 日志
    log_every: int = 5
    eval_ratio: float = 0.1


# =============================================================================
# ========================= DPO 数据集 =========================================
# =============================================================================

class DPODataset(Dataset):
    """DPO 偏好数据集"""
    
    def __init__(self, data: List[Dict], max_len: int, system_prompt: str = ""):
        self.data = data
        self.max_len = max_len
        self.enc = tiktoken.get_encoding("gpt2")
        self.system_prompt = system_prompt
    
    def __len__(self):
        return len(self.data)
    
    def _encode_qa(self, question: str, answer: str) -> List[int]:
        """将 prompt+answer 编码为 token ids"""
        text = ""
        if self.system_prompt:
            text += f"### System:\n{self.system_prompt}\n\n"
        text += f"### User:\n{question}\n\n### Assistant:\n{answer}\n\n"
        
        ids = self.enc.encode(text)
        return ids[:self.max_len]
    
    def _encode_prompt(self, question: str) -> List[int]:
        """只编码 prompt 部分（用于确定 loss 计算起点）"""
        text = ""
        if self.system_prompt:
            text += f"### System:\n{self.system_prompt}\n\n"
        text += f"### User:\n{question}\n\n### Assistant:\n"
        return self.enc.encode(text)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        chosen_text = item["chosen"]
        rejected_text = item["rejected"]
        
        chosen_ids = self._encode_qa(prompt, chosen_text)
        rejected_ids = self._encode_qa(prompt, rejected_text)
        prompt_ids = self._encode_prompt(prompt)
        prompt_len = len(prompt_ids)
        
        return {
            "chosen_ids": chosen_ids,
            "rejected_ids": rejected_ids,
            "prompt_len": prompt_len,
        }


def dpo_collate_fn(batch, max_len, pad_id=0):
    """DPO 专用 collate: 分别 pad chosen 和 rejected"""
    
    chosen_ids_list = [b["chosen_ids"] for b in batch]
    rejected_ids_list = [b["rejected_ids"] for b in batch]
    prompt_lens = [b["prompt_len"] for b in batch]
    
    max_chosen = min(max(len(x) for x in chosen_ids_list), max_len)
    max_rejected = min(max(len(x) for x in rejected_ids_list), max_len)
    max_total = max(max_chosen, max_rejected)
    
    def pad_batch(ids_list, target_len):
        padded = []
        masks = []
        for ids in ids_list:
            ids = ids[:target_len]
            pad_len = target_len - len(ids)
            mask = [1] * len(ids) + [0] * pad_len
            ids = ids + [pad_id] * pad_len
            padded.append(ids)
            masks.append(mask)
        return torch.tensor(padded, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)
    
    chosen_ids, chosen_mask = pad_batch(chosen_ids_list, max_total)
    rejected_ids, rejected_mask = pad_batch(rejected_ids_list, max_total)
    
    return {
        "chosen_ids": chosen_ids,
        "chosen_mask": chosen_mask,
        "rejected_ids": rejected_ids,
        "rejected_mask": rejected_mask,
        "prompt_lens": torch.tensor(prompt_lens, dtype=torch.long),
    }


# =============================================================================
# ========================= DPO 训练器 =========================================
# =============================================================================

class DPOTrainer:
    """DPO 训练器"""
    
    def __init__(self, dpo_config: DPOConfig):
        self.dpo_config = dpo_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_type = "cuda" if self.device.startswith("cuda") else "cpu"
        
        self._load_models()
        self._load_data()
        self._setup_optimizer()
    
    def _load_models(self):
        """加载 policy model (可训练) 和 reference model (冻结)"""
        print(f"[DPO] Loading SFT model from {self.dpo_config.sft_model_path}")
        
        checkpoint = torch.load(self.dpo_config.sft_model_path, map_location="cpu")
        self.model_config = checkpoint["config"]
        self.chat_template = checkpoint.get("chat_template", {})
        
        # Policy model (可训练)
        self.policy_model = GPT(self.model_config)
        state_dict = checkpoint["model"]
        cleaned = {k.replace("module.", "").replace("_orig_mod.", ""): v 
                   for k, v in state_dict.items()}
        self.policy_model.load_state_dict(cleaned, strict=True)
        self.policy_model.to(self.device)
        self.policy_model.set_gradient_checkpointing(False)
        
        # Reference model (冻结, 不需要梯度)
        self.ref_model = GPT(self.model_config)
        self.ref_model.load_state_dict(cleaned, strict=True)
        self.ref_model.to(self.device)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        total_params = sum(p.numel() for p in self.policy_model.parameters())
        print(f"[DPO] Policy model params: {total_params:,}")
        print(f"[DPO] Reference model: frozen copy")
        
        sft_loss = checkpoint.get("val_loss", "unknown")
        print(f"[DPO] SFT val_loss: {sft_loss}")
    
    def _load_data(self):
        """加载 DPO 数据"""
        cfg = self.dpo_config
        print(f"[DPO] Loading data from {cfg.dpo_data_path}")
        
        with open(cfg.dpo_data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        # 过滤掉无效数据
        valid_data = []
        for item in raw_data:
            if (item.get("prompt") and item.get("chosen") and item.get("rejected")
                and item["chosen"].strip() != item["rejected"].strip()):
                valid_data.append(item)
        
        print(f"[DPO] Valid samples: {len(valid_data)} / {len(raw_data)}")
        
        random.seed(42)
        random.shuffle(valid_data)
        
        val_size = max(1, int(len(valid_data) * cfg.eval_ratio))
        val_data = valid_data[:val_size]
        train_data = valid_data[val_size:]
        
        collate = partial(dpo_collate_fn, max_len=cfg.max_seq_len)
        
        self.train_dataset = DPODataset(train_data, cfg.max_seq_len, cfg.system_prompt)
        self.val_dataset = DPODataset(val_data, cfg.max_seq_len, cfg.system_prompt)
        
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=cfg.batch_size,
            shuffle=True, collate_fn=collate, drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=cfg.batch_size,
            shuffle=False, collate_fn=collate,
        )
        
        print(f"[DPO] Train: {len(train_data)}, Val: {len(val_data)}")
    
    def _setup_optimizer(self):
        """优化器"""
        self.optimizer = self.policy_model.configure_optimizers(
            weight_decay=self.dpo_config.weight_decay,
            learning_rate=self.dpo_config.learning_rate,
            device_type=self.device_type,
        )
        
        steps_per_epoch = len(self.train_loader) // self.dpo_config.grad_accum_steps
        self.total_steps = steps_per_epoch * self.dpo_config.num_epochs
        print(f"[DPO] Total steps: {self.total_steps}")
    
    def _get_lr(self, step):
        cfg = self.dpo_config
        if step < cfg.warmup_steps:
            return cfg.learning_rate * (step + 1) / cfg.warmup_steps
        if step >= self.total_steps:
            return cfg.min_lr
        decay_ratio = (step - cfg.warmup_steps) / (self.total_steps - cfg.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

    def _get_log_probs(self, model, input_ids, mask, prompt_lens):
        """
        计算每个 response token 的 log probability
        只计算 assistant 回复部分（prompt 之后的 token）
        """
        logits, _ = model(input_ids)
    
        # shift: logits[t] 预测 input_ids[t+1]
        shift_logits = logits[:, :-1, :]
        shift_ids = input_ids[:, 1:]
        shift_mask = mask[:, 1:].clone()  # ← 关键修复：加 .clone() 避免 inplace
    
        # 计算每个 token 的 log prob
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_ids.unsqueeze(2)).squeeze(2)
    
        # 构建 response mask：prompt 部分设为 False，只保留 response 部分
        B, T = token_log_probs.shape
        # 用非 inplace 方式构建 mask
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        prompt_mask = pos >= (prompt_lens.unsqueeze(1) - 1)  # -1 因为 shift
        response_mask = shift_mask & prompt_mask
    
        # 对有效 token 求和
        masked_log_probs = (token_log_probs * response_mask).sum(dim=1)
        valid_tokens = response_mask.sum(dim=1).clamp(min=1)
    
        # 返回平均 log prob
        return masked_log_probs / valid_tokens

    
    def _compute_dpo_loss(self, batch):
        """
        DPO Loss 计算
        L = -log σ(β · (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))
        """
        chosen_ids = batch["chosen_ids"].to(self.device)
        chosen_mask = batch["chosen_mask"].to(self.device)
        rejected_ids = batch["rejected_ids"].to(self.device)
        rejected_mask = batch["rejected_mask"].to(self.device)
        prompt_lens = batch["prompt_lens"].to(self.device)
        
        # Policy model log probs
        policy_chosen_logps = self._get_log_probs(
            self.policy_model, chosen_ids, chosen_mask, prompt_lens
        )
        policy_rejected_logps = self._get_log_probs(
            self.policy_model, rejected_ids, rejected_mask, prompt_lens
        )
        
        # Reference model log probs (no grad)
        with torch.no_grad():
            ref_chosen_logps = self._get_log_probs(
                self.ref_model, chosen_ids, chosen_mask, prompt_lens
            )
            ref_rejected_logps = self._get_log_probs(
                self.ref_model, rejected_ids, rejected_mask, prompt_lens
            )
        
        # DPO loss
        chosen_rewards = self.dpo_config.beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = self.dpo_config.beta * (policy_rejected_logps - ref_rejected_logps)
        
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
        
        # 统计信息
        with torch.no_grad():
            reward_margin = (chosen_rewards - rejected_rewards).mean().item()
            chosen_reward = chosen_rewards.mean().item()
            rejected_reward = rejected_rewards.mean().item()
            accuracy = (chosen_rewards > rejected_rewards).float().mean().item()
        
        return loss, {
            "reward_margin": reward_margin,
            "chosen_reward": chosen_reward,
            "rejected_reward": rejected_reward,
            "accuracy": accuracy,
        }
    
    @torch.no_grad()
    def evaluate(self):
        """验证集评估"""
        self.policy_model.eval()
        total_loss = 0
        total_acc = 0
        count = 0
        
        for batch in self.val_loader:
            with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                loss, stats = self._compute_dpo_loss(batch)
            total_loss += loss.item()
            total_acc += stats["accuracy"]
            count += 1
        
        self.policy_model.train()
        return total_loss / max(count, 1), total_acc / max(count, 1)
    
    def train(self):
        """主训练循环"""
        cfg = self.dpo_config
        os.makedirs(cfg.output_dir, exist_ok=True)
        log_file = os.path.join(cfg.output_dir, "dpo_log.txt")
        
        print(f"\n{'='*60}")
        print(f"  DPO Training Start")
        print(f"  Beta: {cfg.beta}")
        print(f"  Epochs: {cfg.num_epochs}")
        print(f"  Effective batch: {cfg.batch_size * cfg.grad_accum_steps}")
        print(f"  Learning rate: {cfg.learning_rate}")
        print(f"  Total steps: {self.total_steps}")
        print(f"{'='*60}\n")
        
        self.policy_model.train()
        global_step = 0
        best_val_loss = float("inf")
        
        for epoch in range(cfg.num_epochs):
            self.optimizer.zero_grad()
            epoch_loss = 0
            epoch_acc = 0
            micro_count = 0
            
            for micro_step, batch in enumerate(self.train_loader):
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    loss, stats = self._compute_dpo_loss(batch)
                
                loss_scaled = loss / cfg.grad_accum_steps
                loss_scaled.backward()
                
                epoch_loss += loss.item()
                epoch_acc += stats["accuracy"]
                micro_count += 1
                
                if (micro_step + 1) % cfg.grad_accum_steps == 0:
                    norm = torch.nn.utils.clip_grad_norm_(
                        self.policy_model.parameters(), cfg.grad_clip
                    )
                    
                    lr = self._get_lr(global_step)
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = lr
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    
                    if global_step % cfg.log_every == 0:
                        avg_loss = epoch_loss / micro_count
                        avg_acc = epoch_acc / micro_count
                        print(f"  epoch {epoch+1} step {global_step:4d} | "
                              f"loss {loss.item():.4f} | acc {stats['accuracy']:.2f} | "
                              f"margin {stats['reward_margin']:.3f} | "
                              f"lr {lr:.2e} | norm {norm:.4f}")
                        
                        with open(log_file, "a") as f:
                            f.write(f"{global_step} train loss={loss.item():.4f} "
                                    f"acc={stats['accuracy']:.2f} "
                                    f"margin={stats['reward_margin']:.3f}\n")
            
            # Epoch 结束：验证
            val_loss, val_acc = self.evaluate()
            print(f"\n[Epoch {epoch+1}/{cfg.num_epochs}] "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}")
            
            with open(log_file, "a") as f:
                f.write(f"epoch_{epoch+1} val loss={val_loss:.4f} acc={val_acc:.2f}\n")
            
            # # 保存
            # ckpt_path = os.path.join(cfg.output_dir, f"dpo_epoch_{epoch+1}.pt")
            # torch.save({
            #     "model": self.policy_model.state_dict(),
            #     "config": self.model_config,
            #     "dpo_config": cfg,
            #     "epoch": epoch + 1,
            #     "global_step": global_step,
            #     "val_loss": val_loss,
            #     "val_acc": val_acc,
            #     "chat_template": self.chat_template,
            # }, ckpt_path)
            # print(f"  Saved: {ckpt_path}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(cfg.output_dir, "dpo_best.pt")
                torch.save({
                    "model": self.policy_model.state_dict(),
                    "config": self.model_config,
                    "dpo_config": cfg,
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "chat_template": self.chat_template,
                }, best_path)
                print(f"  Saved best: {best_path}")
        
        print(f"\n[DPO] Training complete! Best val_loss: {best_val_loss:.4f}")


# =============================================================================
# ========================= 入口 ===============================================
# =============================================================================

if __name__ == "__main__":
    config = DPOConfig()
    
    import sys
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, val = arg.split("=", 1)
            key = key.lstrip("-")
            if hasattr(config, key):
                field_type = type(getattr(config, key))
                setattr(config, key, field_type(val))
    
    trainer = DPOTrainer(config)
    trainer.train()