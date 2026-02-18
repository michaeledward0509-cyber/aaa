from pathlib import Path
from typing import Dict, List
import os

HUGGINGFACE_REPO = os.getenv("HUGGINGFACE_REPO", "natix-network-org")
HUGGINGFACE_CACHE_DIR: Path = Path.home() / ".cache" / "huggingface"

# Binary task: roadwork (1) vs no roadwork (0).
# Add more datasets here for training. Then run: poetry run python base_miner/datasets/download_data.py
# See docs/DataSources.md for where to get data and how to use imagefolder: for local paths.
IMAGE_DATASETS: Dict[str, List[Dict[str, str]]] = {
    "Roadwork": [
        {"path": f"{HUGGINGFACE_REPO}/roadwork"},
        # Extra roadwork/construction data (uncomment and run download_data.py):
        # {"path": "LouisChen15/ConstructionSite"},
        # Local folder with train/roadwork/ and train/no_roadwork/ (optional):
        # {"path": "imagefolder:/path/to/your/roadwork_data"},
    ],
}