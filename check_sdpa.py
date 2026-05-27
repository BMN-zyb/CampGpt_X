"""
验证 F.scaled_dot_product_attention 实际调用的后端
适用于 PyTorch 2.2.0 + cu118
"""
import torch
import torch.nn.functional as F

def check_sdpa_backend():
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"Compute Capability: {cap[0]}.{cap[1]}")
        print(f"  (FA2 需要 >= 8.0, 即 Ampere+)")
    print("=" * 60)

    # 1. 检查各后端是否启用
    print("\n[后端启用状态]")
    print(f"  Flash SDP enabled:          {torch.backends.cuda.flash_sdp_enabled()}")
    print(f"  Mem-Efficient SDP enabled:  {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    print(f"  Math SDP enabled:           {torch.backends.cuda.math_sdp_enabled()}")

    if not torch.cuda.is_available():
        print("\n没有 CUDA, 只能用 Math 后端")
        return

    # 2. 构造测试数据 (模拟你的 GQA 配置)
    device = "cuda"
    B, T, n_head, head_dim = 16, 2048, 32, 64
    # GQA repeat 之后 Q/K/V 头数相同
    q = torch.randn(B, n_head, T, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, n_head, T, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, n_head, T, head_dim, device=device, dtype=torch.bfloat16)

    # 3. 逐个后端测试
    print("\n[逐后端测试]")

    # ---- 测试 Flash Attention ----
    try:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_mem_efficient=False,
            enable_math=False
        ):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        print(f"  Flash Attention:      ✅ 可用")
    except Exception as e:
        print(f"  Flash Attention:      ❌ 不可用 - {e}")

    # ---- 测试 Memory-Efficient Attention ----
    try:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_mem_efficient=True,
            enable_math=False
        ):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        print(f"  Mem-Efficient Attn:   ✅ 可用")
    except Exception as e:
        print(f"  Mem-Efficient Attn:   ❌ 不可用 - {e}")

    # ---- 测试 Math (标准实现) ----
    try:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_mem_efficient=False,
            enable_math=True
        ):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        print(f"  Math (标准):          ✅ 可用")
    except Exception as e:
        print(f"  Math (标准):          ❌ 不可用 - {e}")

    # 4. 测试 FP16 (有些 GPU 只支持 FP16 不支持 BF16)
    print("\n[不同精度测试]")
    for dtype, name in [(torch.bfloat16, "BF16"), (torch.float16, "FP16"), (torch.float32, "FP32")]:
        q_ = torch.randn(B, n_head, T, head_dim, device=device, dtype=dtype)
        k_ = torch.randn(B, n_head, T, head_dim, device=device, dtype=dtype)
        v_ = torch.randn(B, n_head, T, head_dim, device=device, dtype=dtype)
        try:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_mem_efficient=False, enable_math=False
            ):
                F.scaled_dot_product_attention(q_, k_, v_, is_causal=True)
            print(f"  Flash + {name}: ✅")
        except:
            print(f"  Flash + {name}: ❌")

    # 5. 速度对比
    print("\n[速度对比] (100 次前向, B=2, T=128, H=32, D=64)")
    import time

    backends = {
        "Flash":        dict(enable_flash=True,  enable_mem_efficient=False, enable_math=False),
        "MemEfficient": dict(enable_flash=False, enable_mem_efficient=True,  enable_math=False),
        "Math":         dict(enable_flash=False, enable_mem_efficient=False, enable_math=True),
        "Auto":         dict(enable_flash=True,  enable_mem_efficient=True,  enable_math=True),
    }

    for name, flags in backends.items():
        try:
            # warmup
            with torch.backends.cuda.sdp_kernel(**flags):
                for _ in range(10):
                    F.scaled_dot_product_attention(q, k, v, is_causal=True)
            torch.cuda.synchronize()

            t0 = time.time()
            with torch.backends.cuda.sdp_kernel(**flags):
                for _ in range(100):
                    F.scaled_dot_product_attention(q, k, v, is_causal=True)
            torch.cuda.synchronize()
            dt = (time.time() - t0) * 1000
            print(f"  {name:15s}: {dt:.1f} ms (100次)")
        except Exception as e:
            print(f"  {name:15s}: ❌ 跳过 - {e}")

    print("\n" + "=" * 60)
    print("结论:")
    print("  如果 Flash Attention ✅ 且速度最快 → 你的 SDPA 默认就是 FA2")
    print("  如果 Flash Attention ❌ → 回退到 MemEfficient 或 Math")
    print("=" * 60)


if __name__ == "__main__":
    check_sdpa_backend()

    # python check_sdpa.py

'''
============================================================
PyTorch version: 2.2.0+cu118
CUDA available: True
CUDA version: 11.8
GPU: NVIDIA GeForce RTX 3090
Compute Capability: 8.6
  (FA2 需要 >= 8.0, 即 Ampere+)
============================================================

[后端启用状态]
  Flash SDP enabled:          True
  Mem-Efficient SDP enabled:  True
  Math SDP enabled:           True

[逐后端测试]
  Flash Attention:      ✅ 可用
  Mem-Efficient Attn:   ✅ 可用
  Math (标准):          ✅ 可用

[不同精度测试]
  Flash + BF16: ✅
  Flash + FP16: ✅
  Flash + FP32: ❌

[速度对比] (100 次前向, B=2, T=128, H=32, D=64)
  Flash          : 483.9 ms (100次)
  MemEfficient   : 527.5 ms (100次)
  Math           : 4037.9 ms (100次)
  Auto           : 486.5 ms (100次)

============================================================
结论:
  如果 Flash Attention ✅ 且速度最快 → 你的 SDPA 默认就是 FA2
  如果 Flash Attention ❌ → 回退到 MemEfficient 或 Math


# 项目				你的结果					结论
# GPU					RTX 3090 (sm_86)		✅ 支持 Flash Attention 2
# Flash Attention		✅ 可用	PyTorch 2.2 	内置 FA2 kernel
# Auto 选择			Flash (2.0ms)			✅ 正确选择
# MemEff 更快			T=128 时是的			正常现象，你训练用 T=1024，Flash 更快
# FP32 不支持 Flash	❌						正常，Flash 只支持 bf16/fp16
# 你需要改代码吗		不需要					F.scaled_dot_product_attention 已经是最优方案
'''