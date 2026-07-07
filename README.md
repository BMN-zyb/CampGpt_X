# 请给我免费的star⭐吧，十分感谢！

---
license: apache-2.0
language:
  - en
tags:
  - text-generation
  - education
  - student-handbook
  - campus-qa
  - custom-architecture
pipeline_tag: text-generation
---

# CampGPT-X · 大学学生手册问答小型 GPT

> 🤗 HuggingFace: https://huggingface.co/bmnzyb/CampGPT_X

一个**从零实现**的中小型 GPT 语言模型项目，完整走通 **预训练 → SFT → DPO** 三阶段训练流程，最终用于「大学学生手册」问答。模型融合了当前主流大模型的多项现代架构改进（GQA / RoPE / RMSNorm / SwiGLU / MoE），代码量适中、注释详尽，非常适合作为**学习 LLM 全链路（数据 → 架构 → 训练 → 对齐 → 部署）的教学项目**。

---

## ✨ 项目亮点

- **完整三阶段流水线**：预训练（FineWeb-Edu 10B tokens）→ SFT 监督微调 → DPO 偏好对齐，一个仓库跑通全流程。
- **现代架构**：Decoder-only Transformer，采用 GQA + RoPE + RMSNorm + SwiGLU + MoE（含共享专家）。
- **工程优化齐全**：BF16 混合精度、FlashAttention（PyTorch SDPA）、TF32、Fused AdamW、DDP / FSDP(ZeRO-2)、梯度累积、梯度检查点、`torch.compile`、KV Cache 推理加速。
- **自动化数据构建**：从 PDF 学生手册出发，调用大模型 API 自动生成 SFT / DPO 训练数据。
- **开箱即用的推理与导出**：命令行 / Flask API 推理服务，一键导出为 HuggingFace 兼容格式。
- **全中文详细注释**：所有 `.py` 源码均已逐块、逐关键行添加简体中文注释。

## 🏗️ 模型规格

| 属性 | 取值 |
|----------|-------|
| 参数量 | 322,680,576 (322.7M) |
| 架构 | Transformer（GQA + RoPE + SwiGLU + MoE） |
| 层数 | 12 |
| 注意力头 | 12（KV 头：4，即 GQA） |
| 隐藏维度 | 768 |
| 上下文长度 | 1024 |
| 分词器 | tiktoken（GPT-2，50257 词表） |
| 训练流程 | Pretrain → SFT → DPO |

> 模型规模可在 [config.py](config.py) 中通过 `get_model_config()` 切换（内置 `CampGPT_X / tiny / small / medium / large / xl / small-dense / GPT-2-124M` 等 8 种预设）。

## 🧬 架构组件一览

| 组件 | 作用 | 替代了什么 |
|------|------|-----------|
| **RMSNorm** | 归一化，去掉均值中心化与 bias，更轻量稳定 | LayerNorm |
| **RoPE** | 旋转位置编码，注入相对位置、支持长度外推 | 绝对位置编码 |
| **GQA** | 分组查询注意力，多个 Q 头共享少量 KV 头，推理省显存 | 标准 MHA |
| **SwiGLU** | SiLU 门控 FFN，表达力更强 | GELU FFN |
| **MoE + 共享专家** | 混合专家，Top-K 稀疏路由，激活参数少、容量大 | 单一 FFN |
| **KV Cache** | 自回归生成时缓存 K/V，避免重复计算 | —（推理优化） |

---

## 📂 项目结构

按「配置 → 架构 → 数据 → 训练 → 评估 → 部署」分层组织：

### 🔧 配置与模型架构
| 文件 | 说明 |
|------|------|
| [config.py](config.py) | **配置中心**。`GPTConfig` dataclass 定义全部超参，`get_model_config()` 提供 8 种规模预设。全项目最上游，被模型和所有训练脚本依赖。 |
| [model.py](model.py) | **模型架构核心（最重要）**。从零实现 Decoder-only Transformer：`RMSNorm` / `RotaryPositionalEmbedding` / `GroupedQueryAttention` / `KVCache` / `SwiGLUFFN` / `MoEGate` / `MoEFFN` / `Block` / `GPT`。被所有训练与推理脚本导入，是唯一的模型定义。 |

### 📊 数据准备
| 文件 | 说明 |
|------|------|
| [fineweb.py](fineweb.py) | **预训练数据预处理**。下载 FineWeb-Edu，用 tiktoken 分词，多进程切分为约 100 个 `uint16` 分片存入 `edu_fineweb10B/`，供预训练读取。 |
| [build_dataset.py](build_dataset.py) | **SFT / DPO 数据构建**。从学生手册 PDF 提取文本→清洗→分块，调用 Qwen3-Max（OpenAI 兼容接口）自动生成问答与偏好对。核心类：`PDFParser`、`QwenClient`、`SFTDataGenerator`、`DPODataGenerator`、`DataQualityChecker`。输出 `sft_dataset.json` 与 `dpo_dataset.json`。 |

### 🎓 三阶段训练
| 文件 | 阶段 | 说明 |
|------|------|------|
| [train_CampGPT_X_plus.py](train_CampGPT_X_plus.py) | ① 预训练 | **预训练主脚本（All-in-One）**。在 FineWeb-Edu ~10B tokens 上做 next-token 预训练。集成分布式（DDP/FSDP）、BF16、FlashAttention、TF32、Fused AdamW、梯度累积/检查点、`torch.compile`、warmup+余弦调度、HellaSwag 评测与 checkpoint 保存。输入 `edu_fineweb10B/*.npy`，输出 `log/model_*.pt`。 |
| [train_sft.py](train_sft.py) | ② SFT | **监督微调**。加载预训练权重，在对话数据上全参数微调。核心：`SFTConfig`、`ChatTokenizer`（按 `### System/User/Assistant` 模板拼接、对非 assistant 部分做 loss mask）、`SFTDataset`、`SFTTrainer`。输入预训练权重 + `sft_dataset.json`，输出 `sft_output/sft_best.pt`。 |
| [train_dpo.py](train_dpo.py) | ③ DPO | **直接偏好优化**。复制 SFT 权重为可训练策略模型 π 与冻结参考模型 π_ref，在偏好对上最小化 DPO 损失。核心：`DPOConfig`、`DPODataset`、`DPOTrainer`、`_get_log_probs`、`_compute_dpo_loss`。输入 `sft_best.pt` + `dpo_dataset.json`，输出 `dpo_output/dpo_best.pt`。 |

### 🧪 评估与测试
| 文件 | 说明 |
|------|------|
| [hellaswag.py](hellaswag.py) | HellaSwag 常识推理评估（completion 风格），对 4 个候选结尾算续写 loss 取最小者，产出 `acc` / `acc_norm`。被预训练脚本在训练过程中调用。 |
| [test_sft.py](test_sft.py) | 加载 `sft_output/sft_best.pt`，用若干学生手册问题测试 SFT 模型的问答效果。`python test_sft.py` 运行。 |
| [test_dpo.py](test_dpo.py) | 通过 `serve.CampGPTServer` 加载最终模型做单轮问答的快速人工验证。 |

### 🚀 部署与导出
| 文件 | 说明 |
|------|------|
| [serve.py](serve.py) | **推理服务**。核心类 `CampGPTServer`：加载权重与对话模板、tiktoken 编解码、调用 `model.generate`（KV Cache + top-k/top-p 采样）。支持命令行交互 / Flask API / 单次问答三种模式，是项目对外的问答入口。 |
| [export_hf.py](export_hf.py) | **模型导出**。`export_to_hf` 将 checkpoint 转为 HuggingFace 兼容目录（`config.json`、权重、tokenizer、对话模板、模型卡 README、上传脚本）。 |

### 🛠️ 工具与辅助
| 文件 | 说明 |
|------|------|
| [check_sdpa.py](check_sdpa.py) | 诊断工具：验证当前 PyTorch 环境下 SDPA 实际调用的后端（FlashAttention2 / 显存高效 / Math），检测 GPU 算力是否满足 FA2，并做多后端基准。`python check_sdpa.py` 运行。 |
| [requirements.txt](requirements.txt) | Python 依赖（PyTorch 2.2.0 + cu118 等）。 |
| `bash.sh` / `upload.sh` | 训练启动命令集合 / 上传 HuggingFace Hub 脚本。 |
| `student-code-of-conduct.pdf` | SFT/DPO 数据构建所用的源学生手册 PDF。 |

---

## 🧭 新人上手路线图

> **强烈建议按下面顺序阅读**，由「配置 → 架构 → 数据 → 训练 → 推理」逐层深入。每一站都标注了阅读目标和该重点看的函数。

### 第 1 站 · 看懂配置（约 10 分钟）
📄 [config.py](config.py)
- 先浏览 `GPTConfig` 的每个字段，建立「这个模型长什么样」的整体印象（层数、维度、GQA 头数、MoE 专家数等）。
- 看一眼 `get_model_config()` 里的 8 种预设，理解不同规模的差异。

### 第 2 站 · 吃透模型架构（核心，值得花最多时间）
📄 [model.py](model.py)
- 这是全项目最重要的文件。建议**按组件从简到繁**阅读：
  1. `RMSNorm` —— 最简单的归一化
  2. `RotaryPositionalEmbedding`（RoPE）+ `rotate_half` / `apply_rotary_pos_emb` —— 位置编码
  3. `GroupedQueryAttention` + `KVCache` —— 注意力与推理缓存（GQA 的 `_repeat_kv`、SDPA 的 `is_causal`）
  4. `SwiGLUFFN` —— 门控前馈网络
  5. `MoEGate` / `MoEFFN` —— Top-K 路由、`index_add_` 散射聚合、`aux_loss` 负载均衡
  6. `Block` —— 把上面组件按「Pre-Norm + 残差」拼成一层
  7. `GPT` —— 顶层模型，重点看 `forward`、`generate`（prefill/decode 两阶段）、`configure_optimizers`
- 读完这一站，你就理解了 GQA + RoPE + RMSNorm + SwiGLU + MoE 是如何组装起来的。

### 第 3 站 · 数据从哪里来
📄 [fineweb.py](fineweb.py) → 📄 [build_dataset.py](build_dataset.py)
- `fineweb.py`：预训练数据是怎么被分词并切成 token 分片的。
- `build_dataset.py`：SFT / DPO 数据是怎么从 PDF 出发、用大模型自动生成的（重点看三类 SFT 数据的 prompt 设计，以及 DPO 的 chosen/rejected 偏好对构造）。

### 第 4 站 · 三阶段训练（按顺序读）
📄 [train_CampGPT_X_plus.py](train_CampGPT_X_plus.py) → 📄 [train_sft.py](train_sft.py) → 📄 [train_dpo.py](train_dpo.py)
- **预训练**：工程优化最集中的文件，重点看 `DataLoaderLite`（分片读取 + x/y 错位切分）、训练步（前向 + 含 MoE `aux_loss` 的 loss + 反向 + step）、分布式初始化、学习率调度。
- **SFT**：重点看 **loss mask**——为什么只在 assistant 回复部分算 loss（非回答部分 label 置 `-100`）。
- **DPO**：重点看 `_get_log_probs`（序列 log 概率）与 `_compute_dpo_loss`（对照偏好优化公式逐步理解 policy vs reference、隐式奖励、`β`）。

### 第 5 站 · 把模型用起来
📄 [serve.py](serve.py) → 📄 [test_sft.py](test_sft.py) / [test_dpo.py](test_dpo.py) → 📄 [export_hf.py](export_hf.py)
- `serve.py`：看 `CampGPTServer.chat` 如何拼 prompt、调 `generate`、抽取回复。
- `test_*.py`：跑起来看看效果。
- `export_hf.py`：把模型导出成 HuggingFace 格式发布。

### 按需 / 排查
📄 [hellaswag.py](hellaswag.py)（预训练期间的常识评估） · 📄 [check_sdpa.py](check_sdpa.py)（排查 FlashAttention 后端与环境）

---

## ⚙️ 环境准备

```bash
# 建议 Python 3.10+，CUDA 11.8 环境
pip install -r requirements.txt

# （可选）确认 FlashAttention 后端是否可用
python check_sdpa.py
```

> FlashAttention2 需要 Ampere 及以上架构（Compute Capability ≥ 8.0）。若不满足，SDPA 会自动回退到显存高效 / Math 后端。

## 🚀 完整流程（从数据到部署）

> 部分脚本中的**数据路径、API Key、权重路径**为示例值，运行前请按源码顶部注释/配置类自行调整。

```bash
# ── 阶段一：预训练 ──────────────────────────────
# 1) 准备预训练数据（下载 FineWeb-Edu 并分词为 token 分片 → edu_fineweb10B/）
python fineweb.py

# 2) 预训练（产出 log/model_*.pt）
#    单卡：
CUDA_VISIBLE_DEVICES=0 python train_CampGPT_X_plus.py
#    多卡（例：2 卡）：
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_CampGPT_X_plus.py

# ── 阶段二：SFT 监督微调 ────────────────────────
# 3) 构建 SFT/DPO 数据集（需在 build_dataset.py 的 DatasetConfig 中填入 API Key）
python build_dataset.py           # 产出 sft_dataset.json、dpo_dataset.json

# 4) SFT（加载预训练权重 → 产出 sft_output/sft_best.pt）
python train_sft.py
python test_sft.py                # 快速测试 SFT 效果

# ── 阶段三：DPO 偏好对齐 ────────────────────────
# 5) DPO（加载 sft_best.pt → 产出 dpo_output/dpo_best.pt）
python train_dpo.py
python test_dpo.py                # 快速测试最终模型

# ── 部署 / 导出 ────────────────────────────────
python serve.py                   # 命令行交互式问答（或作为模块被导入提供 API）
python export_hf.py               # 导出为 HuggingFace 兼容格式
```

## 🔄 训练三阶段流水线

1. **Pretrain（预训练）**：在 FineWeb-Edu 约 10B tokens 上做 next-token 预训练，得到具备通用语言能力的基础模型。
2. **SFT（监督微调）**：在学生手册问答对上做全参数微调，让模型学会「按对话格式回答问题」。
3. **DPO（偏好优化）**：用 chosen（优答）/ rejected（劣答）偏好对进一步对齐，让回答更符合期望。

### 数据与产物速查表

| 阶段 | 脚本 | 输入 | 输出 |
|------|------|------|------|
| 预训练数据 | `fineweb.py` | FineWeb-Edu 原始数据 | `edu_fineweb10B/*.npy` |
| 预训练 | `train_CampGPT_X_plus.py` | `edu_fineweb10B/*.npy` | `log/model_*.pt` |
| 数据构建 | `build_dataset.py` | `student-code-of-conduct.pdf` | `sft_dataset.json`、`dpo_dataset.json` |
| SFT | `train_sft.py` | 预训练权重 + `sft_dataset.json` | `sft_output/sft_best.pt` |
| DPO | `train_dpo.py` | `sft_best.pt` + `dpo_dataset.json` | `dpo_output/dpo_best.pt` |

## 💬 推理用法

```python
from serve import CampGPTServer

server = CampGPTServer("campgpt-student-handbook")
response = server.chat("What are the requirements for a scholarship?")
print(response)
```

### 对话格式

```text
### System:
You are a helpful university assistant...

### User:
What are the scholarship requirements?

### Assistant:
Based on the student handbook...
```

## ⚠️ 局限性

- 模型规模较小，容量有限。
- 知识范围仅限于训练所用的特定学生手册。
- 可能会「幻觉」出训练数据中不存在的细节。

---

## 📖 参考

- RMSNorm: https://arxiv.org/abs/1910.07467
- RoPE: https://arxiv.org/abs/2104.09864
- GQA: https://arxiv.org/abs/2305.13245
- SwiGLU: https://arxiv.org/abs/2002.05202
- DeepSeekMoE: https://arxiv.org/abs/2401.06066
- DPO: https://arxiv.org/abs/2305.18290
- FineWeb-Edu: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu



# 代码部分由AI生成。


## 欢迎贡献代码，提出问题和建议。如果你发现了bug或者有新的功能想法，请提交一个Issue让我知道。你也可以通过Fork项目并提交Pull Request来贡献代码。 如果你喜欢这个项目，欢迎给它一个星星⭐，这是对我最大的支持！


# 如果你觉得我的开源项目对你有帮助，可以赞助我一杯咖啡嘛，十分感谢！！！


