# Getting Roadwork Data for Training

To improve your miner you need **binary-labeled image data**: **roadwork** (label 1) and **no roadwork** (label 0). Below are practical ways to get it and how to plug it into this repo.

---

## 1. Use the existing dataset

The repo already uses **`natix-network-org/roadwork`** on Hugging Face (~8.5k images, 10.5 GB). Download it with:

```bash
poetry run python base_miner/datasets/download_data.py
```

This is a good starting point. To beat competitors you typically need **more and more diverse** data (see options below).

---

## 2. Public datasets (Hugging Face and others)

### Hugging Face

| Dataset | Description | Use as |
|--------|--------------|--------|
| [**natix-network-org/roadwork**](https://huggingface.co/datasets/natix-network-org/roadwork) | Official roadwork dataset (already in config) | Roadwork (label 1) |
| [**LouisChen15/ConstructionSite**](https://huggingface.co/datasets/LouisChen15/ConstructionSite) | Construction site images (10k+), captions and annotations | Roadwork / construction (label 1) |
| [**Programmer-RD-AI/road-issues-detection-dataset**](https://huggingface.co/datasets/Programmer-RD-AI/road-issues-detection-dataset) | Road issues (potholes, etc.); can filter for “work zone” style | Roadwork or negative, depending on filters |
| **Generic street / driving** (e.g. search “driving dataset”, “street view”) | Scenes with no construction | No roadwork (label 0) |

Search for more: [huggingface.co/datasets](https://huggingface.co/datasets) with tags like `image-classification`, `construction`, `road`, `driving`, `street-view`.

### Academic / external

- **VTTI work zone detection** – [Segmentation and detection of work zone scenes](https://github.com/VTTI/Segmentation-and-detection-of-work-zone-scenes): binary “work zone” vs “non–work zone” images (often need to request or scrape).
- **ROADWork (ICCV 2025)** – [ROADWork dataset](https://iccv.thecvf.com/virtual/2025/poster/550): work zone recognition; check if public and if you can export “work zone / no work zone” image lists.

Use these by either (a) converting to Hugging Face or imagefolder format and adding to config, or (b) building a local **imagefolder** (see below).

---

## 3. Create your own dataset (local imagefolder)

You can add **local** image folders without uploading to Hugging Face. The loader supports the **`imagefolder`** format.

### Folder layout (by class)

Put images in subfolders named by class; the folder name becomes the label:

```
/path/to/roadwork_data/
  train/
    roadwork/          # label 1
      img1.jpg
      img2.png
    no_roadwork/       # label 0
      img1.jpg
      img2.png
```

Optional: add `test/` and `validation/` with the same `roadwork/` and `no_roadwork/` subfolders.

### Getting the images

- **Street / dashcam footage**: frame extraction from videos (OpenCV, `ffmpeg`).
- **Google / Bing Street View**: use official APIs or tools (respect ToS and rate limits).
- **Stock / free photos**: search “roadwork”, “construction zone”, “clear road”, “highway” on Unsplash, Pexels, etc.
- **Labeling**: if you have unlabeled images, use [Label Studio](https://labelstud.io/), [CVAT](https://www.cvat.ai/), or a simple script to assign 0/1 and move files into `roadwork/` and `no_roadwork/`.

Then register the path in config (see below) as `imagefolder:/path/to/roadwork_data`.

---

## 4. Adding data to this repo

Training uses **two** dataset groups:

- **“Real”** = no roadwork (label 0)
- **“Fake”** = roadwork (label 1)

`base_miner/config.py` defines `IMAGE_DATASETS`. Right now it only has one category, `"Roadwork"`. To add more sources:

### Option A: Add more Hugging Face datasets

Edit `base_miner/config.py` and add entries under the same or new keys. Example:

```python
IMAGE_DATASETS: Dict[str, List[Dict[str, str]]] = {
    "Roadwork": [
        {"path": f"{HUGGINGFACE_REPO}/roadwork"},
        {"path": "LouisChen15/ConstructionSite"},  # extra roadwork
    ],
    # If you have a "no roadwork" dataset, add it and use it as "real" in training:
    # "NoRoadwork": [
    #     {"path": "some-org/street-view-no-construction"},
    # ],
}
```

Then run:

```bash
poetry run python base_miner/datasets/download_data.py
```

(Use `--force_redownload` if you want to re-download.)

### Option B: Use a local imagefolder

1. Create the folder structure above (e.g. `train/roadwork/`, `train/no_roadwork/`).
2. In config, use the `imagefolder:` prefix and the **directory that contains** `train/` (and optionally `test/`, `validation/`):

```python
# Example: local path
{"path": "imagefolder:/home/user/data/roadwork_data"}
```

The loader in `base_miner/datasets/download_data.py` will use `load_dataset("imagefolder", data_dir=directory)`. Your training script must map these to “real” vs “fake” (e.g. `no_roadwork` → real, `roadwork` → fake) when building `real_datasets` and `fake_datasets`.

---

## 5. Expected format for each image source

Each dataset (HF or imagefolder) should be compatible with `ImageDataset` in `base_miner/datasets/image_dataset.py`. Rows can have:

- **`image`** – PIL Image or bytes, or  
- **`url`** / **`image_url`** – image fetched at runtime  

Optional: **`name`** or **`filename`** for the `id` field.

For **imagefolder**, Hugging Face adds a **`label`** column from the folder names; the rest of the pipeline may still rely on separate “real” vs “fake” dataset lists when building `RealFakeDataset`.

---

## 6. Quick checklist

| Step | Action |
|------|--------|
| 1 | Use existing `natix-network-org/roadwork` (already in config). |
| 2 | Add more **roadwork** sources (e.g. ConstructionSite, your own imagefolder). |
| 3 | Add **no roadwork** sources (street/driving datasets or your own `no_roadwork/` folder). |
| 4 | Update `IMAGE_DATASETS` in `base_miner/config.py` (and any training script that builds real/fake from it). |
| 5 | Run `download_data.py` for HF datasets; use `imagefolder:...` for local data. |
| 6 | Train/fine-tune your model (ViT or other) on the combined data, then submit and point the miner to your model. |

---

## 7. Optional: upload your dataset to Hugging Face

If you build a custom dataset locally:

1. Create a repo on [huggingface.co/datasets](https://huggingface.co/datasets).
2. Use the [imagefolder guide](https://huggingface.co/docs/datasets/image_dataset) or push the same folder structure (e.g. `train/roadwork/`, `train/no_roadwork/`).
3. Add the repo to `IMAGE_DATASETS` as `{"path": "your-username/your-dataset"}`.

That keeps everything in one place and makes reruns and sharing easier.
