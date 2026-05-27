# train_sft.py

"""
SFT (Supervised Fine-Tuning) 全量微调
加载预训练权重 → 在对话数据上训练 → 保存SFT权重
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

os.environ["TIKTOKEN_CACHE_DIR"] = "./tiktoken_cache"
import tiktoken

from model import GPT


# =============================================================================
# ========================= SFT 配置 ==========================================
# =============================================================================

@dataclass
class SFTConfig:
    # 路径
    PREtrained_path: str = "log/model_00999_CampGPT_X.pt"
    SFTtrained_path: str = "sft_output/sft_best.pt"
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
# ========================= SFT 训练器 =========================================
# =============================================================================

class SFTTrainer:
    """SFT 全量微调训练器"""
    
    def __init__(self, sft_config: SFTConfig):
        self.sft_config = sft_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_type = "cuda" if self.device.startswith("cuda") else "cpu"
        
        # 初始化
        # self._load_model_PreTrain()
        self._load_models_SFT()



    def _load_model_PreTrain(self):
        """加载预训练权重"""
        print(f"[SFT] Loading pretrained model from {self.sft_config.PREtrained_path}")
        
        checkpoint = torch.load(self.sft_config.PREtrained_path, map_location="cpu")
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
  
  


    def _load_models_SFT(self):
        print(f"[DPO] Loading SFT model from {self.sft_config.SFTtrained_path}")
        
        checkpoint = torch.load(self.sft_config.SFTtrained_path, map_location="cpu")
        self.model_config = checkpoint["config"]
        self.chat_template = checkpoint.get("chat_template", {})
        
        state_dict = checkpoint["model"]
        cleaned = {k.replace("module.", "").replace("_orig_mod.", ""): v 
                   for k, v in state_dict.items()}

        self.model = GPT(self.model_config)
        self.model.load_state_dict(cleaned, strict=True)
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[DPO] Policy model params: {total_params:,}")
        print(f"[DPO] Reference model: frozen copy")
        
        sft_loss = checkpoint.get("val_loss", "unknown")
        print(f"[DPO] SFT val_loss: {sft_loss}")
    

    
    def _test_generation(self):
        """训练后快速测试生成"""
        print(f"\n{'='*40} Generation Test {'='*40}")
        
        test_questions = [
            "Who is considered an Appellate Officer according to the student handbook?",
            "What does the term 'Associate Dean' refer to in the UHCL Student Handbook?",
            "How is 'Good Standing' defined for students at UHCL?",
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
                        prompt_t, max_new_tokens=64,
                        temperature=0.0, top_k=50, top_p=0.9
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
    trainer._test_generation()