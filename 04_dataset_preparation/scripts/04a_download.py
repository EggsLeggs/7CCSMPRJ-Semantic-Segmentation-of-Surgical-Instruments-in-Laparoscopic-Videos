"""Download LEMON videos from Hugging Face. Requires authentication."""
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="visurg/LEMON",
    repo_type="dataset",
    local_dir="./lemon_data",
)
