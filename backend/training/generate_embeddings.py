"""
Generate ViT frame embeddings from extracted face crops.

Takes each per-video folder of face crops and runs the same Vision Transformer
that the live pipeline uses (from config.VIDEO_MODEL_PRIMARY), then saves the
sequence of 768-dim CLS-token embeddings as a single .npy file per video.

Input layout (from extract_faces.py):
    faces/
    ├── real/
    │   └── 000/000.jpg 001.jpg ...
    └── fake/
        └── 000_Deepfakes/000.jpg 001.jpg ...

Output layout (consumed by ff_dataset.py):
    embeddings/
    ├── real/
    │   └── 000.npy         # shape (T, 768)
    └── fake/
        └── 000_Deepfakes.npy

Runtime:
  ~15-25 min on RTX 4050 for a full FF++ subset.
  ~4-6 HOURS on CPU. Use GPU for this step.

Safe to re-run: skips videos whose .npy already exists.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

# Add parent so we can import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


BATCH_SIZE = 32     # face crops processed at once; adjust if GPU OOM

# Pure vision transformer for EMBEDDINGS (not the classifier used in the live app).
# We deliberately use a different, plain ViT here because the live-app classifier
# is a SigLIP-based model that needs text input when used as a raw backbone.
# 768-dim CLS output matches our LSTM's expected embedding_dim.
EMBEDDING_MODEL = "google/vit-base-patch16-224"


def load_vit(model_name: str):
    """Load ViT for FEATURE extraction (not classification)."""
    processor = AutoImageProcessor.from_pretrained(model_name)
    # AutoModel gives us the raw backbone (no classification head)
    # so we get embeddings, not fake/real logits
    model = AutoModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return processor, model, device


def embed_video_folder(video_dir: Path, processor, model, device) -> np.ndarray:
    """
    Embed all face crops in one video's folder. Returns array of shape (T, D).
    Batches the crops for GPU efficiency.
    """
    crop_paths = sorted(video_dir.glob("*.jpg"))
    if not crop_paths:
        return np.zeros((0, 768), dtype=np.float32)

    all_embeddings = []

    for i in range(0, len(crop_paths), BATCH_SIZE):
        batch_paths = crop_paths[i:i + BATCH_SIZE]
        pil_images = []
        for p in batch_paths:
            img_bgr = cv2.imread(str(p))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(img_rgb))

        if not pil_images:
            continue

        inputs = processor(images=pil_images, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # CLS token = the first position of last_hidden_state
            # Shape: (batch, 768)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]

        all_embeddings.append(cls_embeddings.cpu().numpy())

    if not all_embeddings:
        return np.zeros((0, 768), dtype=np.float32)

    return np.concatenate(all_embeddings, axis=0).astype(np.float32)


def process_split(faces_root: Path, embeddings_root: Path, label: str,
                  processor, model, device):
    """Process every video folder under faces_root/<label>/."""
    src = faces_root / label
    dst = embeddings_root / label
    dst.mkdir(parents=True, exist_ok=True)

    video_folders = sorted([d for d in src.iterdir() if d.is_dir()])
    print(f"\n[{label}] {len(video_folders)} video folders in {src}")

    n_done = 0
    for video_dir in tqdm(video_folders, unit="video"):
        out_file = dst / f"{video_dir.name}.npy"
        if out_file.exists():
            n_done += 1
            continue

        embeddings = embed_video_folder(video_dir, processor, model, device)
        if embeddings.shape[0] > 0:
            np.save(out_file, embeddings)
            n_done += 1

    return n_done


def main():
    parser = argparse.ArgumentParser(description="Generate ViT embeddings from face crops")
    parser.add_argument("--faces-root", required=True,
                        help="Path to faces/ folder (with real/ and fake/ subdirs)")
    parser.add_argument("--embeddings-root", required=True,
                        help="Where to write .npy files (e.g. training/embeddings/)")
    parser.add_argument("--model", default=None,
                        help="HuggingFace model name (default: config.VIDEO_MODEL_PRIMARY)")
    args = parser.parse_args()

    faces_root = Path(args.faces_root)
    embeddings_root = Path(args.embeddings_root)
    model_name = args.model or EMBEDDING_MODEL

    print(f"Loading model: {model_name}")
    processor, model, device = load_vit(model_name)
    print(f"Running on: {device.upper()}")

    n_real = process_split(faces_root, embeddings_root, "real",
                           processor, model, device)
    n_fake = process_split(faces_root, embeddings_root, "fake",
                           processor, model, device)

    print("\n" + "=" * 60)
    print(f"REAL: {n_real} videos embedded")
    print(f"FAKE: {n_fake} videos embedded")
    print(f"Output written to: {embeddings_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()