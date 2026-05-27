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

# CampGPT-Student-Handbook

A compact GPT model trained for university student handbook Q&A.

## Model Details

| Property | Value |
|----------|-------|
| Parameters | 322,680,576 (322.7M) |
| Architecture | Transformer (GQA + RoPE + SwiGLU + MoE) |
| Layers | 12 |
| Heads | 12 (KV: 4) |
| Embedding | 768 |
| Context Length | 1024 |
| Tokenizer | tiktoken (GPT-2, 50257 vocab) |
| Training | Pretrain -> SFT -> DPO |

## Training Pipeline

1. **Pretrain**: 10B tokens from FineWeb-Edu
2. **SFT**: Fine-tuned on student handbook Q&A pairs
3. **DPO**: Preference optimization with chosen/rejected pairs

## Usage

```python
from serve import CampGPTServer

server = CampGPTServer("campgpt-student-handbook")
response = server.chat("What are the requirements for a scholarship?")
print(response)
```

## Chat Format

```text
### System:
You are a helpful university assistant...

### User:
What are the scholarship requirements?

### Assistant:
Based on the student handbook...
```

## Limitations

- Small model with limited capacity
- Knowledge limited to the specific student handbook used for training
- May hallucinate details not in the training data