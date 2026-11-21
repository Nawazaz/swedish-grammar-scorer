"""Upload the model and dataset cards in hub/ to the Hugging Face Hub.

    python upload_cards.py                 # needs a write token (huggingface-cli login)
"""

from huggingface_hub import HfApi

MODEL_ID = "Nawazaz/swedish-grammar-scorer"
DATASET_ID = "Nawazaz/swedish-grammar-data"

api = HfApi()
api.upload_file(path_or_fileobj="hub/model_card.md", path_in_repo="README.md",
                repo_id=MODEL_ID, repo_type="model", commit_message="Add model card")
api.upload_file(path_or_fileobj="hub/dataset_card.md", path_in_repo="README.md",
                repo_id=DATASET_ID, repo_type="dataset", commit_message="Add dataset card")
print(f"Done: https://huggingface.co/{MODEL_ID} and https://huggingface.co/datasets/{DATASET_ID}")
