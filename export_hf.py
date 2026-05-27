# export_hf.py

"""
将训练好的模型导出为 HuggingFace 兼容格式
可以上传到 HuggingFace Hub
"""

import os
import json
import shutil
from typing import Optional

import torch
import tiktoken

from config import GPTConfig, get_model_config
from model import GPT
from train_dpo import DPOConfig


# =============================================================================
# ========================= README 生成 ========================================
# =============================================================================

def generate_readme(config_dict, model_config, model_name):
    """生成 README.md 内容，避免 f-string 嵌套反引号的问题"""
    total_params = config_dict["total_params"]
    moe_str = " + MoE" if model_config.use_moe else ""

    lines = [
        "---",
        "license: apache-2.0",
        "language:",
        "  - en",
        "tags:",
        "  - text-generation",
        "  - education",
        "  - student-handbook",
        "  - campus-qa",
        "  - custom-architecture",
        "pipeline_tag: text-generation",
        "---",
        "",
        f"# {model_name}",
        "",
        "A compact GPT model trained for university student handbook Q&A.",
        "",
        "## Model Details",
        "",
        "| Property | Value |",
        "|----------|-------|",
        f"| Parameters | {total_params:,} ({total_params/1e6:.1f}M) |",
        f"| Architecture | Transformer (GQA + RoPE + SwiGLU{moe_str}) |",
        f"| Layers | {model_config.n_layer} |",
        f"| Heads | {model_config.n_head} (KV: {model_config.n_kv_head}) |",
        f"| Embedding | {model_config.n_embd} |",
        f"| Context Length | {model_config.block_size} |",
        "| Tokenizer | tiktoken (GPT-2, 50257 vocab) |",
        "| Training | Pretrain -> SFT -> DPO |",
        "",
        "## Training Pipeline",
        "",
        "1. **Pretrain**: 10B tokens from FineWeb-Edu",
        "2. **SFT**: Fine-tuned on student handbook Q&A pairs",
        "3. **DPO**: Preference optimization with chosen/rejected pairs",
        "",
        "## Usage",
        "",
        "```python",
        "from serve import CampGPTServer",
        "",
        'server = CampGPTServer("campgpt-student-handbook")',
        'response = server.chat("What are the requirements for a scholarship?")',
        "print(response)",
        "```",
        "",
        "## Chat Format",
        "",
        "```text",
        "### System:",
        "You are a helpful university assistant...",
        "",
        "### User:",
        "What are the scholarship requirements?",
        "",
        "### Assistant:",
        "Based on the student handbook...",
        "```",
        "",
        "## Limitations",
        "",
        "- Small model with limited capacity",
        "- Knowledge limited to the specific student handbook used for training",
        "- May hallucinate details not in the training data",
    ]

    return "\n".join(lines)


# =============================================================================
# ========================= 上传脚本生成 ========================================
# =============================================================================

def generate_upload_script(model_name, output_dir):
    """生成上传到 HuggingFace Hub 的 bash 脚本"""
    lines = [
        "#!/bin/bash",
        "# Upload to HuggingFace Hub",
        "# pip install huggingface_hub",
        "",
        "python -c \"",
        "from huggingface_hub import HfApi, create_repo",
        "",
        "api = HfApi()",
        f"repo_id = 'YOUR_USERNAME/{model_name}'",
        "",
        "create_repo(repo_id, exist_ok=True, repo_type='model')",
        "",
        "api.upload_folder(",
        f"    folder_path='{output_dir}',",
        "    repo_id=repo_id,",
        "    repo_type='model',",
        ")",
        "print(f'Uploaded to https://huggingface.co/{repo_id}')",
        "\"",
    ]
    return "\n".join(lines)


# =============================================================================
# ========================= 主导出函数 =========================================
# =============================================================================

def export_to_hf(
    checkpoint_path: str = "dpo_output/dpo_best.pt",
    output_dir: str = "campgpt-student-handbook",
    model_name: str = "CampGPT-Student-Handbook",
):
    """
    导出模型为 HuggingFace 兼容格式

    输出结构:
    output_dir/
    ├── config.json          # 模型配置
    ├── model.safetensors    # 权重 (safetensors 格式)
    ├── pytorch_model.bin    # 权重 (PyTorch 格式, 备用)
    ├── tokenizer.json       # Tokenizer 信息
    ├── chat_template.json   # 对话模板
    ├── README.md            # 模型卡片
    ├── upload.sh            # 上传脚本
    ├── model.py             # 模型定义 (方便复现)
    └── config.py            # 配置定义
    """

    print(f"[Export] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model_config = checkpoint["config"]
    chat_template = checkpoint.get("chat_template", {})
    state_dict = checkpoint["model"]

    # 清理 key（去掉 DDP/FSDP/compile 可能添加的前缀）
    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("_orig_mod.", "")
        cleaned[k] = v

    os.makedirs(output_dir, exist_ok=True)

    # ==================== 1. 保存模型配置 ====================
    config_dict = {
        "model_type": "campgpt",
        "architectures": ["CampGPT"],
        "vocab_size": model_config.vocab_size,
        "n_embd": model_config.n_embd,
        "n_head": model_config.n_head,
        "n_kv_head": model_config.n_kv_head,
        "n_layer": model_config.n_layer,
        "block_size": model_config.block_size,
        "norm_eps": model_config.norm_eps,
        "multiple_of": model_config.multiple_of,
        "use_moe": model_config.use_moe,
        "n_experts": getattr(model_config, "n_experts", 0),
        "n_experts_per_tok": getattr(model_config, "n_experts_per_tok", 0),
        "n_shared_experts": getattr(model_config, "n_shared_experts", 0),
        "total_params": sum(p.numel() for p in cleaned.values()),
        "training_stages": ["pretrain_10B", "sft", "dpo"],
        "val_loss": checkpoint.get("val_loss", None),
    }

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"  Saved config.json")



    # ==================== 2. 保存权重 ====================
    # 处理共享权重：wte.weight 和 lm_head.weight 是同一个 tensor
    # PyTorch 格式支持共享，直接保存
    torch.save(cleaned, os.path.join(output_dir, "pytorch_model.bin"))
    size_mb = sum(v.numel() * v.element_size() for v in cleaned.values()) / 1e6
    print(f"  Saved pytorch_model.bin ({size_mb:.1f} MB)")

    # safetensors 不支持共享 tensor，需要去重
    try:
        from safetensors.torch import save_file

        # 找出共享权重，只保留一份
        safetensors_dict = {}
        seen_data_ptrs = {}
        shared_keys = {}  # 记录共享关系

        for k, v in cleaned.items():
            data_ptr = v.data_ptr()
            if data_ptr in seen_data_ptrs:
                # 这个 tensor 和之前的某个 key 共享内存，跳过
                shared_keys[k] = seen_data_ptrs[data_ptr]
                print(f"  [safetensors] Skip shared: {k} -> {seen_data_ptrs[data_ptr]}")
            else:
                seen_data_ptrs[data_ptr] = k
                safetensors_dict[k] = v

        save_file(safetensors_dict, os.path.join(output_dir, "model.safetensors"))
        print(f"  Saved model.safetensors")

        # 保存共享权重映射关系（加载时需要恢复）
        if shared_keys:
            with open(os.path.join(output_dir, "shared_weights.json"), "w") as f:
                json.dump(shared_keys, f, indent=2)
            print(f"  Saved shared_weights.json: {shared_keys}")

    except ImportError:
        print(f"  [Skip] safetensors not installed, skipping")






    
    # ==================== 3. Tokenizer 信息 ====================
    tokenizer_info = {
        "type": "tiktoken",
        "encoding": "gpt2",
        "vocab_size": 50257,
        "special_tokens": {
            "pad_token": "<|endoftext|>",
            "eos_token": "<|endoftext|>",
        },
    }
    with open(os.path.join(output_dir, "tokenizer.json"), "w") as f:
        json.dump(tokenizer_info, f, indent=2)
    print(f"  Saved tokenizer.json")

    # ==================== 4. 对话模板 ====================
    chat_info = {
        "system_prompt": chat_template.get("system_prompt", ""),
        "user_prefix": chat_template.get("user_prefix", "\n\n### User:\n"),
        "assistant_prefix": chat_template.get("assistant_prefix", "\n\n### Assistant:\n"),
        "turn_end": chat_template.get("turn_end", "\n\n"),
        "template": "### System:\n{system}\n\n### User:\n{user}\n\n### Assistant:\n{assistant}\n\n",
    }
    with open(os.path.join(output_dir, "chat_template.json"), "w") as f:
        json.dump(chat_info, f, ensure_ascii=False, indent=2)
    print(f"  Saved chat_template.json")

    # ==================== 5. 复制源码 ====================
    for src_file in ["model.py", "config.py"]:
        if os.path.exists(src_file):
            shutil.copy(src_file, os.path.join(output_dir, src_file))
            print(f"  Copied {src_file}")

    # ==================== 6. 生成 README.md ====================
    readme = generate_readme(config_dict, model_config, model_name)
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(readme)
    print(f"  Saved README.md")

    # ==================== 7. 上传脚本 ====================
    upload_script = generate_upload_script(model_name, output_dir)
    with open(os.path.join(output_dir, "upload.sh"), "w") as f:
        f.write(upload_script)
    print(f"  Saved upload.sh")

    print(f"\n[Export] Done! Files saved to {output_dir}/")
    print(f"  To upload: edit upload.sh with your HF username, then run it")


# =============================================================================
# ========================= 入口 ===============================================
# =============================================================================

if __name__ == "__main__":
    import sys

    ckpt = sys.argv[1] if len(sys.argv) > 1 else "dpo_output/dpo_best.pt"
    out = sys.argv[2] if len(sys.argv) > 2 else "campgpt-student-handbook"

    export_to_hf(checkpoint_path=ckpt, output_dir=out)