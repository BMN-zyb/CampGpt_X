torchrun --standalone --nproc_per_node=2 train_gpt2.py

torchrun --standalone --nproc_per_node=2 train_gpt2_simple.py

torchrun --standalone --nproc_per_node=2 train_CampGPT_X.py

python train_CampGPT_X.py


CUDA_VISIBLE_DEVICES=1 python train_CampGPT_X_plus.py
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_CampGPT_X_plus.py

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_CampGPT_X_plus.py


# 按顺序执行
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_CampGPT_X_plus.py
CUDA_VISIBLE_DEVICES=1 python train_sft.py
CUDA_VISIBLE_DEVICES=1 python train_dpo.py
CUDA_VISIBLE_DEVICES=1 python export_hf.py
# cli single api
CUDA_VISIBLE_DEVICES=1 python serve.py --mode cli
CUDA_VISIBLE_DEVICES=1 python serve.py --mode single
CUDA_VISIBLE_DEVICES=1 python serve.py --mode api
CUDA_VISIBLE_DEVICES=1 python test_sft.py
CUDA_VISIBLE_DEVICES=1 python test_dpo.py




export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli upload-large-folder bmnzyb/CampGPT_X campgpt-student-handbook/ --repo-type=model
huggingface-cli upload-large-folder bmnzyb/campgpt campgpt-student-handbook/ --repo-type=model




# 清除镜像设置
unset HF_ENDPOINT
unset HF_MIRROR

# 重新上传
bash upload.sh
bash campgpt-student-handbook/upload.sh
