"""
FaceForensics++ dataset loader for LSTM temporal training.

Assumes embeddings have already been generated (via generate_embeddings.py)
and saved as .npy files in a structured directory:

    embeddings/
    ├── real/
    │   ├── 000.npy         # shape (T, D) — T frames of D-dim embeddings
    │   ├── 001.npy
    │   └── ...
    └── fake/
        ├── 000_Deepfakes.npy
        ├── 001_FaceSwap.npy
        └── ...

Each .npy file = one video's frame-level embeddings.
Label is derived from the parent folder (real=0, fake=1).

Sequences of variable length are handled by cropping / padding to a fixed
SEQ_LEN inside __getitem__.  This keeps the DataLoader simple.
"""
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# Target sequence length after crop/pad. 30 frames ~= 1 second at 30fps.
# Chosen so training batches fit easily on a 4050 (~6GB VRAM headroom).
SEQ_LEN = 30


class FFEmbeddingsDataset(Dataset):
    """
    Serves (embedding_sequence, label) pairs from a directory of .npy files.
    """

    def __init__(self,
                 embeddings_root: str,
                 seq_len: int = SEQ_LEN,
                 split: str = "train",
                 train_ratio: float = 0.8,
                 seed: int = 42):
        """
        Args:
            embeddings_root : path to embeddings/ folder (with real/ and fake/ subdirs)
            seq_len         : all sequences cropped or zero-padded to this length
            split           : "train" or "val"
            train_ratio     : fraction of data used for training
            seed            : RNG seed for the split (must be same across splits)
        """
        self.embeddings_root = Path(embeddings_root)
        self.seq_len = seq_len
        self.split = split

        real_dir = self.embeddings_root / "real"
        fake_dir = self.embeddings_root / "fake"

        if not real_dir.exists() or not fake_dir.exists():
            raise FileNotFoundError(
                f"Expected {real_dir} and {fake_dir} to exist. "
                f"Run generate_embeddings.py first."
            )

        # Collect (file_path, label) tuples
        real_files = sorted(real_dir.glob("*.npy"))
        fake_files = sorted(fake_dir.glob("*.npy"))
        items = [(p, 0) for p in real_files] + [(p, 1) for p in fake_files]

        if len(items) == 0:
            raise RuntimeError(
                f"No .npy files found in {real_dir} or {fake_dir}."
            )

        # Deterministic shuffle + split (so train/val don't leak across sessions)
        rng = np.random.default_rng(seed)
        indices = np.arange(len(items))
        rng.shuffle(indices)

        cutoff = int(len(items) * train_ratio)
        if split == "train":
            keep = indices[:cutoff]
        elif split == "val":
            keep = indices[cutoff:]
        else:
            raise ValueError(f"split must be 'train' or 'val', got {split}")

        self.items = [items[i] for i in keep]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        embeddings = np.load(path)          # shape (T, D)

        # Handle sequence length: crop if too long, zero-pad if too short
        T, D = embeddings.shape
        if T >= self.seq_len:
            # Random crop during training, centre crop during val
            if self.split == "train":
                start = np.random.randint(0, T - self.seq_len + 1)
            else:
                start = (T - self.seq_len) // 2
            seq = embeddings[start:start + self.seq_len]
        else:
            # Zero-pad at the end
            pad = np.zeros((self.seq_len - T, D), dtype=embeddings.dtype)
            seq = np.concatenate([embeddings, pad], axis=0)

        return (
            torch.from_numpy(seq).float(),
            torch.tensor(label, dtype=torch.float32),
        )

    def label_distribution(self):
        """Return (n_real, n_fake) counts for the current split."""
        n_real = sum(1 for _, y in self.items if y == 0)
        n_fake = sum(1 for _, y in self.items if y == 1)
        return n_real, n_fake


# ============================================================
# Quick sanity check: `python ff_dataset.py <embeddings_root>`
# ============================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ff_dataset.py <path/to/embeddings/>")
        print("Skipping test — no path provided.")
        sys.exit(0)

    root = sys.argv[1]
    train_ds = FFEmbeddingsDataset(root, split="train")
    val_ds   = FFEmbeddingsDataset(root, split="val")

    print(f"Train samples: {len(train_ds)}  (real={train_ds.label_distribution()[0]}, fake={train_ds.label_distribution()[1]})")
    print(f"Val samples:   {len(val_ds)}    (real={val_ds.label_distribution()[0]}, fake={val_ds.label_distribution()[1]})")

    x, y = train_ds[0]
    print(f"\nSample shape: {x.shape}, label: {y.item()}")