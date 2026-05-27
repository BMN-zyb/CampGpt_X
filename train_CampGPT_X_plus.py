'''
下面是整合所有优化后的完整代码。我会将其拆分为清晰的模块，并在关键处加上注释。

改进版 GPT 架构 - 完整训练代码
==========================================
架构改进:
  - RMSNorm (替代 LayerNorm)
  - RoPE (替代绝对位置编码)
  - GQA (替代标准 MHA)
  - SwiGLU (替代 GELU FFN)
  - MoE + 共享专家 (替代单一 FFN)

训练优化:
  - BF16 混合精度
  - Flash Attention v2 (通过 PyTorch SDPA)
  - FSDP / DDP 分布式训练 (ZeRO-2)
  - Gradient Checkpointing
  - 梯度累积
  - Fused AdamW
  - TF32 MatMul
  - torch.compile (可选)

推理优化:
  - KV Cache
  - Top-K / Top-P 采样

用法:
  单卡:   CUDA_VISIBLE_DEVICES=1 python train_CampGPT_X_plus.py
  多卡:   CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_CampGPT_X_plus.py


============================================================================
改进版 GPT 架构 — 完整训练代码 (All-in-One)
============================================================================
架构层改进:  RMSNorm · RoPE · GQA · SwiGLU · MoE+共享专家
计算层加速:  BF16混合精度 · FlashAttention v2 · TF32 · Fused AdamW · torch.compile
编译层加速：
并行/分布层加速：   DDP / FSDP(ZeRO-2) · 延迟梯度同步(no_sync)
显存优化层:  梯度累积 · Gradient Checkpointing · 权重共享
推理加速层:  KV Cache

运行方式:
  单卡:  python train.py
  多卡:  torchrun --nproc_per_node=N train.py
============================================================================

'''

import os
import math
import time
import inspect
import functools

import numpy as np
import torch
from torch.nn import functional as F


from config import get_model_config
from model import GPT, KVCache, Block

# ==================== 验证 Flash Attention 后端 ====================
def check_flash_attention(config, device):
    """检测当前环境 F.scaled_dot_product_attention 使用的后端"""
    if device == "cpu":
        print("[Flash Attention] CPU 模式，使用 math 后端")
        return

    print(f"\n{'='*60}")
    print(f"  Flash Attention 后端检测")
    print(f"  PyTorch 版本: {torch.__version__}")
    print(f"  CUDA 版本: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        cap = torch.cuda.get_device_capability()
        print(f"  Compute Capability: {cap[0]}.{cap[1]}")
    print(f"{'='*60}")

    # 构造测试输入
    B, T, n_head, n_kv_head = 2, 128, config.n_head, config.n_kv_head
    head_dim = config.n_embd // config.n_head
    q = torch.randn(B, n_head, T, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, n_head, T, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, n_head, T, head_dim, device=device, dtype=torch.bfloat16)

    backends = {
        "Flash SDP (Flash Attention 2)": dict(enable_flash=True, enable_math=False, enable_mem_efficient=False),
        "Memory-Efficient SDP":          dict(enable_flash=False, enable_math=False, enable_mem_efficient=True),
        "Math SDP (标准实现)":            dict(enable_flash=False, enable_math=True, enable_mem_efficient=False),
    }

    for name, flags in backends.items():
        try:
            with torch.backends.cuda.sdp_kernel(**flags):
                _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            print(f"  ✅ {name}: 可用")
        except RuntimeError as e:
            print(f"  ❌ {name}: 不可用 ({e})")

    # 实际调度测试：让 PyTorch 自动选择
    print(f"\n  自动调度测试:")
    with torch.no_grad():
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"  ✅ F.scaled_dot_product_attention 正常工作")

    # PyTorch 2.2+ 可以查看具体选择了哪个后端
    if hasattr(torch.backends.cuda, 'flash_sdp_enabled'):
        print(f"\n  当前默认开关状态:")
        print(f"    flash_sdp_enabled:          {torch.backends.cuda.flash_sdp_enabled()}")
        print(f"    mem_efficient_sdp_enabled:   {torch.backends.cuda.mem_efficient_sdp_enabled()}")
        print(f"    math_sdp_enabled:            {torch.backends.cuda.math_sdp_enabled()}")

    print(f"{'='*60}\n")



# =============================================================================
# ========================= 设备检测 ==========================================
# =============================================================================

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"[Init] Detected device: {device}")



# =============================================================================
# ========================= FSDP 封装 =========================================
# =============================================================================

def setup_fsdp(model, device):
    """
    FSDP (ZeRO-2 语义): 分片梯度和优化器状态，参数不分片。
    显存大幅降低，通信开销可控。
    """
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
        MixedPrecision,
        BackwardPrefetch,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    bf16_policy = MixedPrecision( # 全模型使用 BF16 混合精度
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    auto_wrap = functools.partial( # 自动包装 Transformer 层，指定 Block 作为层类型
        transformer_auto_wrap_policy,
        transformer_layer_cls={Block},
    )
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,  # 分片梯度和优化器状态，参数不分片，属于 ZeRO-2 级别
        mixed_precision=bf16_policy, # 参数、梯度、buffer 都用 bfloat16
        auto_wrap_policy=auto_wrap, # 自动包装 Transformer 层，指定 Block 作为层类型
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE, # 反向预取，重叠通信和计算
        device_id=torch.cuda.current_device(), # 指定当前 GPU 设备
        limit_all_gathers=True, # 限制 all-gather 的范围，减少通信开销 控制 FSDP gather 时的显存开销
        use_orig_params=True, # 保持原始参数结构，方便访问和优化器配置
    )
    return model


# =============================================================================
# ========================= 数据加载 ==========================================
# =============================================================================

os.environ["TIKTOKEN_CACHE_DIR"] = "./tiktoken_cache"
import tiktoken

def load_tokens(filename):
    npt = np.load(filename)
    return torch.tensor(npt.astype(np.int32), dtype=torch.long)


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, data_root="edu_fineweb10B"):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in ('train', 'val')

        shards = sorted([
            os.path.join(data_root, s)
            for s in os.listdir(data_root) if split in s
        ])
        
        self.shards = shards
        assert len(self.shards) > 0, f"No shards found for split={split} in {data_root}"
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y


# =============================================================================
# ========================= HellaSwag 评估 ====================================
# =============================================================================

from hellaswag import render_example, iterate_examples


def get_most_likely_row(tokens, mask, logits):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_tokens = tokens[..., 1:].contiguous()
    shift_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_tokens.view(-1),
        reduction='none'
    ).view(tokens.size(0), -1)
    shift_mask = mask[..., 1:].contiguous()
    masked = shift_losses * shift_mask
    avg = masked.sum(dim=1) / shift_mask.sum(dim=1)
    return avg.argmin().item()


# =============================================================================
# ========================= 学习率调度 ========================================
# =============================================================================

def get_lr(step, max_lr=6e-4, min_lr=6e-5, warmup_steps=715, max_steps=19073):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# =============================================================================
# ========================= 吞吐量基准测试 ====================================
# =============================================================================

def benchmark(model, config, device, device_type, B=4, T=1024,
              num_steps=50, warmup=10, label="test"):
    """测量训练吞吐量"""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dummy_x = torch.randint(0, config.vocab_size, (B, T), device=device)
    dummy_y = torch.randint(0, config.vocab_size, (B, T), device=device)

    for _ in range(warmup):
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, loss = model(dummy_x, dummy_y)
        loss.backward(); opt.step(); opt.zero_grad()

    if device_type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(num_steps):
        if device_type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        opt.zero_grad()
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, loss = model(dummy_x, dummy_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if device_type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)

    avg = sum(times) / len(times)
    tps = B * T / avg
    peak = torch.cuda.max_memory_allocated() / 1024**3 if device_type == "cuda" else 0
    print(f"[{label}]  {avg*1000:.1f} ms/step | {tps:,.0f} tok/s | {peak:.2f} GB peak")
    return dict(label=label, ms=avg*1000, tps=tps, gb=peak)


# =============================================================================
# ========================= 分布式初始化 =======================================
# =============================================================================

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "DDP requires CUDA"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(2025)
if torch.cuda.is_available():
    torch.cuda.manual_seed(2025)

# TF32 加速矩阵乘法，原理是允许在矩阵乘法中使用 TF32 精度，这是一种介于 FP16 和 FP32 之间的数值格式，能够提供更高的性能，同时保持足够的数值稳定性。TF32 在 NVIDIA Ampere 架构及以上的 GPU 上得到支持，可以显著加速训练过程中的矩阵乘法操作，尤其是在大规模模型训练中。
torch.set_float32_matmul_precision('high')  # # 训练技巧：降显存，提token吞吐量

enc = tiktoken.get_encoding("gpt2")


# =============================================================================
# =========================   主 训 练 循 环   =================================
# =============================================================================

def main():
    # ==================== 超参数 ====================
    # "CampGPT_X": GPTConfig(
    #     n_embd=768, n_head=12, n_kv_head=4, n_layer=12,
    #     use_moe=True, n_shared_experts=1, n_experts=3, n_experts_per_tok=1,
    # ),
    
    model_size = "CampGPT_X"           # "GPT-2-124M"、large、tiny、medium
    total_batch_size = 524288     # ~0.5M tokens tokens 跑10B大概需要多少step？ 524288 tokens/step * 19073 steps ≈ 10B tokens
    B = 32                         # micro batch size
    
    # total_batch_size = 4*1024*96*2  # 524288     # ~0.5M tokens
    # B = 96                          # micro batch size
    
    T = 1024                      # sequence length
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 100
    max_steps = 1000 # 19073
    val_every = 100    # 验证的频率，每多少步评估一次验证集损失
    hella_every = 100  # HellaSwag 评测的频率，评测较慢，所以不需要每次都评测
    gen_every = 100    # 使用 KV Cache 生成的频率，也就是多少步评估一次生成质量
    save_every = 100  # 保存模型的频率，每多少步保存一次
    log_dir = "log"

    
    # ==================== 配置 ====================
    config = get_model_config(model_size)
    # 多卡时自动启用 FSDP
    use_fsdp = ddp and ddp_world_size > 1 and True  # 训练技巧：降显存，降token吞吐量
    use_fused_adamw = True
    use_gradient_checkpointing = True  # 训练技巧：降显存，降token吞吐量
    use_compile = False  # 训练技巧：显存不变，提token吞吐量

    config.use_fsdp = use_fsdp
    config.use_fused_adamw = use_fused_adamw
    config.use_gradient_checkpointing = use_gradient_checkpointing
    config.use_compile = use_compile

    assert total_batch_size % (B * T * ddp_world_size) == 0
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

    if master_process:
        print(f"\n{'='*60}")
        print(f"Training Configuration")
        print(f"{'='*60}")
        print(f"  Model: {model_size}")
        print(f"  Total batch: {total_batch_size:,} tokens")
        print(f"  Micro batch: B={B}, T={T}")
        print(f"  Grad accum steps: {grad_accum_steps}")
        print(f"  World size: {ddp_world_size}")
        print(f"  One epoch need {10000000000/total_batch_size:.0f} steps")
        print(f"  FSDP (ZeRO-2): {use_fsdp}")
        # 梯度检查点：启用后会在前向传播时丢弃中间激活，反向传播时重新计算以节省内存，但会增加计算时间
        print(f"  Use fused AdamW: {use_fused_adamw}")
        print(f"  Gradient Checkpointing: {config.use_gradient_checkpointing}")
        print(f"  torch.compile: {config.use_compile}")

        print(f"{'='*60}\n")

    # ==================== 数据 ====================
    train_loader = DataLoaderLite(B, T, ddp_rank, ddp_world_size, 'train')
    val_loader = DataLoaderLite(B, T, ddp_rank, ddp_world_size, 'val')
    if master_process:
        print(f"Train shards: {len(train_loader.shards)}, Val shards: {len(val_loader.shards)}")

    # ==================== 模型 ====================
    model = GPT(config)
    # model = model.to(torch.bfloat16) # 将模型参数转换为 BF16 精度，减少显存占用。
    model.to(device)



    # ==================== 验证 Flash Attention 后端 ====================
    # if master_process:
    #     check_flash_attention(config, device)



    # ==================== 分布式包装 ====================
    if use_fsdp: # FSDP 包装模型，分片梯度和优化器状态，参数不分片
        model = setup_fsdp(model, device)
        raw_model = model  # FSDP: 直接引用
    elif ddp:  # DDP 只包装模型，不分片参数，只是简单的梯度同步
        model = DDP(model, device_ids=[ddp_local_rank])
        raw_model = model.module
    else:
        raw_model = model

    if master_process:
        print("=== PARAM DTYPE CHECK ===")
        for name, p in raw_model.named_parameters():
            print(name, p.dtype)
            break


    # torch.compile   # 训练技巧：显存不变，提token吞吐量
    if config.use_compile:
        if master_process:
            print("[Compile] Compiling model with torch.compile ...")
        model = torch.compile(model)


    # ==================== 优化器 ====================
    if use_fsdp:
        if use_fused_adamw:  # # 训练技巧：显存不变，提token吞吐量
            # FSDP 下用 use_orig_params=True，可以直接按名称分组
            param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
            decay = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {'params': decay, 'weight_decay': 0.1},
                {'params': nodecay, 'weight_decay': 0.0},
            ]
            fused_ok = 'fused' in inspect.signature(torch.optim.AdamW).parameters # 检查当前环境是否支持 fused AdamW
            optimizer = torch.optim.AdamW(
                optim_groups, lr=max_lr, betas=(0.9, 0.95), eps=1e-8,
                fused=(fused_ok and device_type == "cuda")
            )
            if master_process:
                print(f"  Optimizer: AdamW (FSDP mode, fused={fused_ok})")
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), eps=1e-8)
    else:
        if use_fused_adamw:
            optimizer = raw_model.configure_optimizers(0.1, max_lr, device_type, master_process)
        else:
            optimizer = torch.optim.AdamW(raw_model.parameters(), lr=max_lr, betas=(0.9, 0.95), eps=1e-8)
        

    # ==================== 日志 ====================
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log_{model_size}.txt")
    if master_process:
        with open(log_file, "w") as f:
            pass

    # ==================== KV Cache 分析 ====================
    if master_process and config.use_moe:
        gqa_bytes, mha_bytes = KVCache.memory_footprint(config)
        print(f"\n[KV Cache] GQA: {gqa_bytes/1024**2:.1f} MB, "
              f"MHA: {mha_bytes/1024**2:.1f} MB, "
              f"Savings: {(1-gqa_bytes/mha_bytes)*100:.0f}%\n")


    # ==================== 训练 ====================
    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # ---------- 验证 ----------
        if step % val_every == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_steps = 20
                for _ in range(val_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        _, loss = model(x, y)
                    val_loss_accum += loss.detach() / val_steps
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"step {step:5d} | val loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")

            
            # 保存模型检查点
            if step > 0 and (step % save_every == 0 or last_step):
                ckpt = {
                    'model': raw_model.state_dict() if not use_fsdp else None,
                    'config': config,
                    'step': step,
                    'val_loss': val_loss_accum.item(),
                }
                
                if use_fsdp:
                    from torch.distributed.fsdp import (
                        FullyShardedDataParallel as FSDP,
                        FullStateDictConfig,
                        StateDictType,
                    )
                    cfg_save = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg_save):
                        ckpt['model'] = model.state_dict()
                if master_process:
                    path = os.path.join(log_dir, f"model_{step:05d}_{model_size}.pt")
                    torch.save(ckpt, path)
                    print(f"  Saved checkpoint: {path}")




        # ---------- HellaSwag ----------
        if (step % hella_every == 0 or last_step) and not config.use_compile:
        # if (step % hella_every == 0 or last_step):
            model.eval()
            num_correct = 0
            num_total = 0
            for i, example in enumerate(iterate_examples("val")):
                if i >= 100: # 评测前100个样本，正式训练时可以去掉这个限制
                    break

                if i % ddp_world_size != ddp_rank:
                    continue
                _, tokens, mask, label = render_example(example)
                tokens, mask = tokens.to(device), mask.to(device)
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, _ = model(tokens)
                    pred = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct += int(pred == label)
            if ddp:
                stats = torch.tensor([num_correct, num_total], dtype=torch.long, device=device)
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                num_correct, num_total = stats.tolist()
            if master_process and num_total > 0:
                acc = num_correct / num_total
                print(f"step {step:5d} | HellaSwag: {num_correct}/{num_total} = {acc:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} hella {acc:.4f}\n")





        # ---------- 文本生成 (使用 KV Cache) ---------- # 训练技巧：增显存，提token吞吐量
        # if (step % gen_every == 0 or last_step) and not config.use_compile:
        if (step % gen_every == 0 or last_step):
            model.eval()
            # 推理时关闭 checkpoint
            if hasattr(raw_model, 'set_gradient_checkpointing'):
                raw_model.set_gradient_checkpointing(False)

            num_seqs = 4
            max_gen_len = 64
            prompt = enc.encode("Hello, I'm a language model,")
            prompt_t = torch.tensor(prompt, dtype=torch.long, device=device)
            prompt_t = prompt_t.unsqueeze(0).repeat(num_seqs, 1)

            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)

            # 用 KV Cache 生成
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    if use_fsdp:
                        # FSDP 模式: 临时聚合全部参数后再生成
                        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                        with FSDP.summon_full_params(model, writeback=False, recurse=True):
                            generated = raw_model.generate(
                                prompt_t, max_new_tokens=max_gen_len - len(prompt),
                                temperature=0.8, top_k=50
                            )
                    else:
                        generated = raw_model.generate(
                            prompt_t, max_new_tokens=max_gen_len - len(prompt),
                            temperature=0.8, top_k=50
                        )

            if master_process:
                for i in range(num_seqs):
                    text = enc.decode(generated[i].tolist())
                    print(f"  sample {i}: {text}")

            # 恢复 checkpoint
            if hasattr(raw_model, 'set_gradient_checkpointing'):
                raw_model.set_gradient_checkpointing(config.use_gradient_checkpointing)





        # ---------- 训练步骤 ----------
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            # DDP 延迟梯度同步: 只在最后一个 micro_step 同步
            if ddp and not use_fsdp: # FSDP 内部已经处理了同步，无需再设置
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
                # model.require_forward_param_sync = True  # 每个 micro_step 都需要同步最新参数

            # 可以是 torch.float16、 torch.bfloat16、 torch.float32
            # 确保训练时走 Flash 而非 Math 后端, 训练循环中你已经有 torch.autocast(dtype=torch.bfloat16)，这保证了 Q/K/V 是 BF16，Flash 后端会被自动选中。
            # 如果你用 dtype=torch.float32，则 Flash 不可用（如检测结果所示），会回退到 Math 后端，速度慢 6 倍。
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                _, loss = model(x, y)
            
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = get_lr(step, max_lr, min_lr, warmup_steps, max_steps)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        optimizer.step()

        if device_type == "cuda":
            torch.cuda.synchronize()

        t1 = time.time()
        dt = t1 - t0
        tps = total_batch_size / dt  # 整个 global batch 的吞吐量

        if master_process:
            print(f"step {step:5d} | loss {loss_accum.item():.6f} | "
                  f"lr {lr:.2e} | norm {norm:.4f} | "
                  f"{dt*1000:.0f}ms | {tps:,.0f} tok/s")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

    # ==================== 清理 ====================
    if ddp:
        destroy_process_group()
    print("\n" + "="*30 + " Done " + "="*30)









# =============================================================================
# ========================= 入口 ===============================================
# =============================================================================

if __name__ == "__main__":
    main()












'''

# 项目				你的结果					结论
# GPU					RTX 3090 (sm_86)		✅ 支持 Flash Attention 2
# Flash Attention		✅ 可用	PyTorch 2.2 	内置 FA2 kernel
# Auto 选择			Flash (2.0ms)			✅ 正确选择
# MemEff 更快			T=128 时是的			正常现象，你训练用 T=1024，Flash 更快
# FP32 不支持 Flash	❌						正常，Flash 只支持 bf16/fp16
# 你需要改代码吗		不需要					F.scaled_dot_product_attention 已经是最优方案


## 问题分析

### 错误本质

`RuntimeError: 'weight' must be 2-D` 发生在 `raw_model.generate()` 中调用 `self.transformer.wte(idx)` 时，说明 embedding 层的 weight 不再是 2D 张量。

### 根本原因：FSDP 对参数的分片

当使用 FSDP 时，模型参数会被**分片（sharded）**。FSDP 会将参数展平为 1D 张量并分散到各个 GPU 上。只有在 FSDP 管理的 `forward()` 调用中，参数才会被临时聚合（all-gather）恢复为原始形状。

### 问题链条

1. 在 train.py 中，`raw_model` 在 FSDP 模式下被设置为 `raw_model = model`（即 FSDP 包装后的模型本身）
2. 但 `generate()` 方法内部直接访问 `self.transformer.wte(idx)`——这是直接访问 FSDP 内部的**子模块**
3. 这种直接访问绕过了 FSDP 的参数聚合机制，此时 `wte.weight` 仍然是**分片后的 1D 扁平张量**，不是原始的 `(vocab_size, n_embd)` 2D 形状
4. `nn.Embedding` 要求 weight 必须是 2D，所以报错

### 为什么训练时的 `forward()` 没问题？

因为训练时调用的是 `model(x, y)`，走的是 FSDP 包装的 `__call__`，FSDP 会在进入 forward 前自动 all-gather 恢复参数，forward 结束后再释放。而 `generate()` 中手动逐层调用子模块，完全绕过了这个机制。

### 总结

**核心问题**：在 FSDP 模式下，`generate()` 方法直接访问被 FSDP 分片的子模块参数，绕过了 FSDP 的参数聚合流程，导致参数形状不正确。

需要解决的方向是：在推理/生成时，要么让参数恢复到未分片状态，要么通过 FSDP 提供的上下文管理器来临时聚合参数，要么在生成前将模型转换为非 FSDP 的完整模型。

'''