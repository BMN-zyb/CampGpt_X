"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm

# ------------------------------------------
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT" # 10 BT tokens 的样本数据集
shard_size = int(1e8) # 100M tokens per shard, total of 100 shards  每个分片一亿个token，总共100个分片

# create the cache the local directory if it doesn't exist yet  创建本地缓存目录（如果尚不存在）
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
# fw = load_dataset("/disk3/Datasets/zengyb/CampGPT_X/edu_fineweb10B/sample/10BT", name=remote_name, split="train")
fw = load_dataset("parquet", data_dir="/disk3/Datasets/zengyb/CampGPT_X/edu_fineweb10B/sample/10BT", split="train")

# # fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train", cache_dir="./hf_cache")   # 👈 关键

# HF_HOME=/disk3/Datasets/zengyb/CampGPT_X/hf_cachenew 
# huggingface-cli download   --repo-type dataset   --resume-download   HuggingFaceFW/fineweb-edu   --local-dir /disk3/Datasets/zengyb/CampGPT_X/edu_fineweb10B   --include "sample/10BT/*"


# HF_HOME=/disk3/Datasets/zengyb/CampGPT_X/hf_cachenew \
# huggingface-cli download \
#   --repo-type dataset \
#   --resume-download \
#   HuggingFaceFW/fineweb-edu \
#   --local-dir /disk3/Datasets/zengyb/CampGPT_X/edu_fineweb10B \
#   --include "sample/10BT/*"


# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token
def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens 将单个文档标记化并返回 uint16 令牌的 numpy 数组
    tokens = [eot] # the special <|endoftext|> token delimits all documents 特殊的<|endoftext|>标记分隔所有文档
    tokens.extend(enc.encode_ordinary(doc["text"]))  # encode_ordinary 不对特殊标记进行编码
    tokens_np = np.array(tokens)  # array 转换为 numpy 数组
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16" # 确保所有标记都适合 uint16
    tokens_np_uint16 = tokens_np.astype(np.uint16) # 转换为 uint16
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder) 将所有文档标记化并写入输出分片，每个分片大小为 shard_size 个标记（最后一个分片有余数）
nprocs = max(1, os.cpu_count()//2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard  预分配缓冲区以保存当前分片
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)  # 
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, fw, chunksize=16): # 并行标记化文档

        # is there enough space in the current shard for the new tokens?  当前分片中是否有足够的空间容纳新标记？
        if token_count + len(tokens) < shard_size:
            # simply append tokens to current shard  只需将标记附加到当前分片
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one 写入当前分片并启动一个新分片
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            # split the document into whatever fits in this shard; the remainder goes to next one 将文档拆分为适合此分片的内容；其余内容转到下一个分片
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc 使用当前文档的剩余部分填充下一个分片
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    # write any remaining tokens as the last shard 写入任何剩余的标记作为最后一个分片
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])
