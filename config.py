from dataclasses import dataclass


# =============================================================================
# ========================= 模型配置 ==========================================
# =============================================================================

@dataclass # 使用 dataclass 来定义模型配置，方便管理和传递参数
class GPTConfig:
    # --- 基础架构 ---
    block_size: int = 1024
    vocab_size: int = 50304 #   50257     # 对齐到 128 的倍数
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

    # --- GQA ---
    n_kv_head: int = 4

    # --- RMSNorm ---
    norm_eps: float = 1e-6

    # --- SwiGLU ---
    multiple_of: int = 64

    # --- MoE ---
    use_moe: bool = True
    n_experts: int = 8
    n_experts_per_tok: int = 2
    n_shared_experts: int = 1
    aux_loss_coeff: float = 0.01  # MoE 辅助损失的权重，通常设置为一个较小的值，如 0.01，来平衡主损失和专家负载平衡损失

    # --- 训练优化选项 ---
    # 梯度检查点：启用后会在前向传播时丢弃中间激活，反向传播时重新计算以节省内存，但会增加计算时间
    use_gradient_checkpointing: bool = False  # 训练技巧：降显存，降token吞吐量
    use_fsdp: bool = False
    use_compile: bool = False


# =============================================================================
# ========================= 预设模型配置 =======================================
# =============================================================================

def get_model_config(model_size: str = "small") -> GPTConfig:
    configs = {
        "CampGPT_X": GPTConfig(
            n_embd=768, n_head=12, n_kv_head=4, n_layer=12,
            use_moe=True, n_shared_experts=1, n_experts=3, n_experts_per_tok=1,
        ),
        "tiny": GPTConfig(
            n_embd=768, n_head=6, n_kv_head=2, n_layer=6,
            use_moe=True, n_shared_experts=1, n_experts=4, n_experts_per_tok=2,
        ),
        "small": GPTConfig(
            n_embd=768, n_head=12, n_kv_head=4, n_layer=12,
            use_moe=True, n_shared_experts=1, n_experts=8, n_experts_per_tok=2,
        ),
        "medium": GPTConfig(
            n_embd=1024, n_head=16, n_kv_head=4, n_layer=24,
            use_moe=False, n_shared_experts=1, n_experts=8, n_experts_per_tok=2,
        ),
        "large": GPTConfig(
            n_embd=1536, n_head=24, n_kv_head=4, n_layer=30,
            use_moe=False, n_shared_experts=1, n_experts=8, n_experts_per_tok=2,
            multiple_of=128,
        ),
        "xl": GPTConfig(
            n_embd=2048, n_head=32, n_kv_head=8, n_layer=36,
            use_moe=True, n_shared_experts=1, n_experts=16, n_experts_per_tok=2,
            multiple_of=128,
        ),
        "small-dense": GPTConfig(
            n_embd=768, n_head=12, n_kv_head=4, n_layer=12,
            use_moe=False,
        ),
        "GPT-2-124M": GPTConfig(
            n_embd=768, n_head=12, n_kv_head=12, n_layer=12,
            use_moe=False,
        ),
    }
    assert model_size in configs, f"Unknown model_size: {model_size}. Choose from {list(configs.keys())}"
    return configs[model_size]





















