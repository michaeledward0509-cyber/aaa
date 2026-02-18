import os
from datasets import load_dataset
from huggingface_hub import login


# Login using e.g. `huggingface-cli login` to access this dataset
# Or set HUGGINGFACE_TOKEN environment variable
token = os.getenv("HUGGINGFACE_TOKEN")
if token:
    login(token=token)
ds = load_dataset("LouisChen15/ConstructionSite")