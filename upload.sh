

python -c "
from huggingface_hub import HfApi, create_repo

api = HfApi()
repo_id =

create_repo(repo_id, exist_ok=True, repo_type='model')

api.upload_folder(
    folder_path='campgpt-student-handbook',
    repo_id=repo_id,
    repo_type='model',
)
print(f'Uploaded to https://huggingface.co/{repo_id}')
"