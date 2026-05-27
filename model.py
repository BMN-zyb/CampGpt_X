
import inspect
import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from config import GPTConfig





# =============================================================================
# ========================= 基础组件: RMSNorm ==================================
# =============================================================================

class RMSNorm(nn.Module):
    """
    RMSNorm: 去掉 mean centering 和 bias，
    只保留 RMS 归一化 + 可学习缩放因子 γ。
    参考: https://arxiv.org/abs/1910.07467
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# =============================================================================
# ========================= 基础组件: RoPE ====================================
# =============================================================================

class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE (Rotary Positional Embedding):
    将相对位置信息通过旋转变换注入 Q/K，支持长度外推。
    参考: https://arxiv.org/abs/2104.09864
    """
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len: int):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        return (
            self.cos_cached[:seq_len].to(x.dtype),
            self.sin_cached[:seq_len].to(x.dtype),
        )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# =============================================================================
# ========================= GQA 注意力 (支持 KV Cache) =========================
# =============================================================================

class GroupedQueryAttention(nn.Module):
    """
    GQA (Grouped-Query Attention):
    Q heads 分组共享 K/V heads，推理时大幅减少 KV Cache 显存。  # 训练技巧：降显存，提token吞吐量
    训练时正常计算，推理时支持 KV Cache。
    参考: https://arxiv.org/abs/2305.13245
    """
    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0

        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep = self.n_head // self.n_kv_head # 每个 KV head 需要被重复的次数
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd

        self.wq = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=False)
        self.wo.NANOGPT_SCALE_INIT = 1

        self.rotary_emb = RotaryPositionalEmbedding(
            self.head_dim, max_seq_len=config.block_size,
        )

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        B, n_kv_head, T, head_dim = x.shape
        x = x[:, :, None, :, :].expand(B, n_kv_head, self.n_rep, T, head_dim)
        return x.reshape(B, n_kv_head * self.n_rep, T, head_dim)

    def forward(self, x, kv_cache=None, start_pos: int = 0):
        """
        Args:
            x: (B, T, C)
            kv_cache: KVCache 对象 (推理时使用, 训练时为 None)
            start_pos: 当前 token 在完整序列中的起始位置 (推理用)
        """
        B, T, C = x.size()

        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # RoPE (位置偏移处理)
        if kv_cache is not None:
            cos, sin = self.rotary_emb(q, start_pos + T)
            cos = cos[start_pos:start_pos + T]
            sin = sin[start_pos:start_pos + T]
        else:
            cos, sin = self.rotary_emb(q, T)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # KV Cache
        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, k, v)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)


        # # 标准注意力计算: att = softmax((q @ k^T) / sqrt(d_k)) @ v
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # k_len = k.size(-2)
        # mask = torch.tril(
        #     torch.ones(T, k_len, device=x.device, dtype=torch.bool)
        # ).view(1, 1, T, k_len)
        # att = att.masked_fill(~mask, torch.finfo(att.dtype).min)
        # att = F.softmax(att, dim=-1)
        # y = att @ v  # 计算加权和，得到注意力输出（B, nh, T, T）x (B, nh, T, hs) -> (B, nh, T, hs)


        # Flash Attention (is_causal: 推理decode阶段T=1时不需要causal mask)  # 训练技巧：降显存，提token吞吐量
        is_causal = True if kv_cache is None else (T > 1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.wo(y)
        return y


# =============================================================================
# ========================= KV Cache ==========================================
# =============================================================================

class KVCache:
    """
    KV Cache: 自回归生成时缓存已计算的 K/V。
    预分配显存，避免反复 cat 造成碎片化。

    GQA 优势: 缓存大小 = n_kv_head * head_dim (而非 n_head * head_dim)
    """
    def __init__(self, config, batch_size: int, max_seq_len: int,
                 device: torch.device, dtype=torch.bfloat16):
        self.n_layers = config.n_layer
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.max_seq_len = max_seq_len

        # 预分配: (n_layers, B, n_kv_head, max_seq_len, head_dim)
        self.k_cache = torch.zeros(
            self.n_layers, batch_size, self.n_kv_head, max_seq_len, self.head_dim,
            device=device, dtype=dtype
        )
        self.v_cache = torch.zeros(
            self.n_layers, batch_size, self.n_kv_head, max_seq_len, self.head_dim,
            device=device, dtype=dtype
        )
        self.seq_len = 0

    def update(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor):
        """写入新 K/V 并返回完整缓存"""
        new_len = k_new.size(2)
        end = self.seq_len + new_len
        self.k_cache[layer_idx, :, :, self.seq_len:end, :] = k_new
        self.v_cache[layer_idx, :, :, self.seq_len:end, :] = v_new
        return (
            self.k_cache[layer_idx, :, :, :end, :],
            self.v_cache[layer_idx, :, :, :end, :],
        )

    def advance(self, n: int = 1):
        self.seq_len += n

    def reset(self):
        self.seq_len = 0
        self.k_cache.zero_()
        self.v_cache.zero_()

    @staticmethod
    def memory_footprint(config, batch_size=1, seq_len=2048, dtype=torch.bfloat16):
        """分析 KV Cache 显存占用"""
        bpe = 2 if dtype in (torch.bfloat16, torch.float16) else 4
        head_dim = config.n_embd // config.n_head
        gqa = 2 * config.n_layer * batch_size * config.n_kv_head * seq_len * head_dim * bpe
        mha = 2 * config.n_layer * batch_size * config.n_head * seq_len * head_dim * bpe
        return gqa, mha


# =============================================================================
# ========================= SwiGLU FFN ========================================
# =============================================================================

class SwiGLUFFN(nn.Module):
    """
    SwiGLU FFN: SiLU 门控 + 3 个权重矩阵
    SwiGLU(x) = (SiLU(xW1) ⊙ xW3) W2
    参考: https://arxiv.org/abs/2002.05202
    """
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(4 * config.n_embd * 2 / 3)
        if hasattr(config, 'multiple_of'):
            hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.w2.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# =============================================================================
# ========================= MoE 层 ============================================
# =============================================================================

class MoEGate(nn.Module):
    """MoE 路由门控: Top-K 路由 + 辅助负载均衡损失"""
    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k = config.n_experts_per_tok
        self.gate = nn.Linear(config.n_embd, config.n_experts, bias=False)
        self.aux_loss_coeff = config.aux_loss_coeff

    def forward(self, x):
        logits = self.gate(x)
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        if self.training:
            probs = F.softmax(logits, dim=-1)
            mask = F.one_hot(indices, num_classes=self.n_experts).sum(dim=1)
            f = mask.float().mean(dim=0)
            P = probs.mean(dim=0)
            aux_loss = self.aux_loss_coeff * self.n_experts * (f * P).sum()
        else:
            aux_loss = torch.tensor(0.0, device=x.device)

        return weights, indices, aux_loss


class MoEFFN(nn.Module):
    """
    MoE FFN: 共享专家 + 路由专家
    参考: DeepSeekMoE (https://arxiv.org/abs/2401.06066)
    """
    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k = config.n_experts_per_tok
        self.n_embd = config.n_embd

        self.gate = MoEGate(config)
        self.experts = nn.ModuleList([SwiGLUFFN(config) for _ in range(config.n_experts)])

        self.n_shared_experts = getattr(config, 'n_shared_experts', 1)
        if self.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [SwiGLUFFN(config) for _ in range(self.n_shared_experts)]
            )

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)

        # 共享专家
        shared_output = torch.zeros_like(x_flat)
        if self.n_shared_experts > 0:
            for se in self.shared_experts:
                shared_output = shared_output + se(x_flat)
            if self.n_shared_experts > 1:
                shared_output = shared_output / self.n_shared_experts

        # 路由专家
        weights, indices, aux_loss = self.gate(x_flat)
        routed_output = torch.zeros_like(x_flat)

        for i in range(self.n_experts):
            token_idx, slot_idx = torch.where(indices == i)
            if token_idx.numel() == 0:
                continue
            expert_input = x_flat[token_idx]
            expert_output = self.experts[i](expert_input)
            expert_weights = weights[token_idx, slot_idx].unsqueeze(-1)
            # routed_output.index_add_(0, token_idx, expert_output * expert_weights)
            contrib = (expert_output * expert_weights).to(routed_output.dtype)
            routed_output.index_add_(0, token_idx, contrib)

        output = (shared_output + routed_output).view(B, T, C)
        return output, aux_loss


# =============================================================================
# ========================= Transformer Block ==================================
# =============================================================================

class Block(nn.Module):
    """
    Transformer Block:
    Pre-RMSNorm → GQA → 残差
    Pre-RMSNorm → SwiGLU (MoE) → 残差
    支持 Gradient Checkpointing
    """
    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.attn = GroupedQueryAttention(config, layer_idx=layer_idx)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.norm_eps)

        self.use_moe = config.use_moe
        if self.use_moe:
            self.ffn = MoEFFN(config)
        else:
            self.ffn = SwiGLUFFN(config)

        self.use_checkpoint = config.use_gradient_checkpointing

    def _attn_block(self, x):
        """注意力子块 (可被 checkpoint 包裹)"""
        return self.attn(self.ln_1(x))

    def _ffn_block(self, x):
        """FFN 子块 (可被 checkpoint 包裹)"""
        if self.use_moe:
            return self.ffn(self.ln_2(x))
        else:
            return self.ffn(self.ln_2(x)), torch.tensor(0.0, device=x.device)

    def forward(self, x, kv_cache=None, start_pos: int = 0):
        """
        前向传播，支持三种模式:
        1. 训练 + checkpoint: 使用 gradient checkpointing 节省显存
        2. 训练 - checkpoint: 标准前向
        3. 推理 (kv_cache is not None): 使用 KV Cache
        """
        # --- 注意力 ---
        if self.use_checkpoint and self.training and kv_cache is None:
            attn_out = torch.utils.checkpoint.checkpoint(
                self._attn_block, x,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        else:
            attn_out = self.attn(self.ln_1(x), kv_cache=kv_cache, start_pos=start_pos)
        x = x + attn_out

        # --- FFN ---
        if self.use_checkpoint and self.training and kv_cache is None:
            if self.use_moe:
                ffn_out, aux_loss = torch.utils.checkpoint.checkpoint(
                    self._ffn_block, x,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                def _dense_ffn(x_in):
                    out = self.ffn(self.ln_2(x_in))
                    return out, torch.tensor(0.0, device=x_in.device)
                ffn_out, aux_loss = torch.utils.checkpoint.checkpoint(
                    _dense_ffn, x,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
        else:
            if self.use_moe:
                ffn_out, aux_loss = self.ffn(self.ln_2(x))
            else:
                ffn_out = self.ffn(self.ln_2(x))
                aux_loss = torch.tensor(0.0, device=x.device)

        x = x + ffn_out
        return x, aux_loss


# =============================================================================
# ========================= GPT 模型 ==========================================
# =============================================================================

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config, layer_idx=i) for i in range(config.n_layer)]),
            ln_f=RMSNorm(config.n_embd, eps=config.norm_eps),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # 权重共享

        self.apply(self._init_weights)
        self._print_model_info()

    # -------------------------------------------------------------------------
    def _init_weights(self, module):  # 权重初始化: Linear 正态分布, Embedding 正态分布, bias 置零
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # -------------------------------------------------------------------------
    def _print_model_info(self):  # 模型参数统计和配置信息展示
        cfg = self.config
        total = sum(p.numel() for p in self.parameters())
        print(f"\n{'='*60}")
        print(f"  Layers={cfg.n_layer}, Heads={cfg.n_head}, KV Heads={cfg.n_kv_head}")
        print(f"  Embed={cfg.n_embd}, BlockSize={cfg.block_size}")
        if cfg.use_moe:
            print(f"  MoE: {cfg.n_experts} experts, top-{cfg.n_experts_per_tok}, "
                  f"{cfg.n_shared_experts} shared")
        print(f"  GradCheckpoint={cfg.use_gradient_checkpointing}")
        print(f"  Total Parameters: {total:,} ({total/1e6:.1f}M)")
        if cfg.use_moe:
            active = self._estimate_active_params()
            print(f"  Active Parameters/token: ~{active:,} ({active/1e6:.1f}M)")
        print(f"{'='*60}\n")

    # -------------------------------------------------------------------------
    def _estimate_active_params(self):  # 估算每 token 活跃参数量 (MoE 模型)
        cfg = self.config
        emb = cfg.vocab_size * cfg.n_embd
        hd = cfg.n_embd // cfg.n_head
        attn = cfg.n_embd * cfg.n_head * hd + cfg.n_embd * cfg.n_kv_head * hd * 2 + cfg.n_head * hd * cfg.n_embd
        norms = cfg.n_embd * 2
        hidden = int(4 * cfg.n_embd * 2 / 3)
        hidden = cfg.multiple_of * ((hidden + cfg.multiple_of - 1) // cfg.multiple_of)
        ffn_single = 3 * cfg.n_embd * hidden
        if cfg.use_moe:
            active_ffn = cfg.n_experts_per_tok * ffn_single + cfg.n_shared_experts * ffn_single
            active_ffn += cfg.n_embd * cfg.n_experts  # gate
        else:
            active_ffn = ffn_single
        per_layer = attn + norms + active_ffn
        return emb + per_layer * cfg.n_layer + cfg.n_embd

    # -------------------------------------------------------------------------
    def forward(self, idx, targets=None, kv_cache=None, start_pos: int = 0):
        B, T = idx.size()
        assert T <= self.config.block_size, \
            f"Sequence length {T} exceeds block_size {self.config.block_size}"

        x = self.transformer.wte(idx)

        total_aux_loss = torch.tensor(0.0, device=idx.device)
        for block in self.transformer.h:
            x, aux_loss = block(x, kv_cache=kv_cache, start_pos=start_pos)
            total_aux_loss = total_aux_loss + aux_loss

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            if self.config.use_moe:
                loss = loss + total_aux_loss

        return logits, loss

    # -------------------------------------------------------------------------
    def configure_optimizers(self, weight_decay, learning_rate, device_type, master_process=True):
        # 只对需要梯度更新的参数进行优化器配置，区分权重衰减和非权重衰减参数

        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        if master_process:
            nd = sum(p.numel() for p in decay_params)
            nn_ = sum(p.numel() for p in nodecay_params)
            print(f"  Decayed tensors: {len(decay_params)}, params: {nd:,}")
            print(f"  Non-decayed tensors: {len(nodecay_params)}, params: {nn_:,}")

        # Fused AdamW 的原理是将多个小的内核调用合并为一个大内核，减少内核启动和内存访问开销，提升训练速度。
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"  Fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate,
            betas=(0.9, 0.95), eps=1e-8, fused=use_fused  # 训练技巧：显存不变，提token吞吐量
        )
        return optimizer

    # -------------------------------------------------------------------------
    def set_gradient_checkpointing(self, enabled: bool):
        """训练/推理切换时动态开关 checkpointing，推理时关闭以节省计算开销"""
        for block in self.transformer.h:
            block.use_checkpoint = enabled

    # -------------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=50, top_p=0.9):
        """
        使用 KV Cache 的高效自回归生成
        """
        B, prompt_len = idx.shape
        total_len = prompt_len + max_new_tokens
        assert total_len <= self.config.block_size

        # 推理模式: 关闭 checkpoint
        self.set_gradient_checkpointing(False)
        self.eval()

        kv_cache = KVCache(
            self.config, B, total_len, idx.device, dtype=torch.bfloat16
        )

        # --- Prefill: 处理整个 prompt ---
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x, _ = block(x, kv_cache=kv_cache, start_pos=0)
        kv_cache.advance(prompt_len)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x[:, -1, :])
        next_token = self._sample(logits, temperature, top_k, top_p)
        generated = [next_token]

        # --- Decode: 逐 token 生成 ---
        for step in range(1, max_new_tokens):
            x = self.transformer.wte(next_token.unsqueeze(1))
            pos = prompt_len + step - 1
            for block in self.transformer.h:
                x, _ = block(x, kv_cache=kv_cache, start_pos=pos)
            kv_cache.advance(1)

            x = self.transformer.ln_f(x)
            logits = self.lm_head(x[:, -1, :])
            next_token = self._sample(logits, temperature, top_k, top_p)
            generated.append(next_token)

        generated = torch.stack(generated, dim=1)
        return torch.cat([idx, generated], dim=1)

    # -------------------------------------------------------------------------
    def _sample(self, logits, temperature, top_k, top_p):
        # 转为 float32 避免 bfloat16 精度问题
        logits = logits.float()

        if temperature == 0:
            return logits.argmax(dim=-1)
        logits = logits / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum_probs > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            indices_to_remove = remove.scatter(1, sorted_idx, remove)
            logits[indices_to_remove] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        # 安全检查：将可能的 nan/inf 替换为 0，避免 multinomial 崩溃
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        # 如果某行全为 0（极端情况），给均匀分布
        zero_rows = (probs.sum(dim=-1) == 0)
        if zero_rows.any():
            probs[zero_rows] = 1.0 / probs.size(-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)