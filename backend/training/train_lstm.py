"""
Train the temporal LSTM on FaceForensics++ ViT embeddings.

Pipeline stage 3 of 3 (after extract_faces.py + generate_embeddings.py):
    face crops  →  ViT embeddings  →  LSTM temporal classifier  ← WE ARE HERE

Produces:
    training/models/lstm_temporal.pt   — trained model weights
    training/models/lstm_metrics.json  — final metrics (acc / prec / rec / f1 / auc)

Usage:
    python training/train_lstm.py --embeddings-root training/embeddings

Runtime on RTX 4050: ~3-5 minutes for a full FF++ subset (~1000 videos).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Make sibling imports work when run as `python training/train_lstm.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ff_dataset import FFEmbeddingsDataset
from lstm_model import TemporalLSTM


# ============================================================
# Metrics helpers (no sklearn dependency — keep training env slim)
# ============================================================
def binary_metrics(logits: np.ndarray, labels: np.ndarray, threshold: float = 0.5):
    probs = 1.0 / (1.0 + np.exp(-logits))       # sigmoid
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    n = len(labels)
    acc  = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    # Rough ROC-AUC via rank statistics (Mann-Whitney U)
    pos = probs[labels == 1]
    neg = probs[labels == 0]
    if len(pos) and len(neg):
        combined = np.concatenate([pos, neg])
        ranks = combined.argsort().argsort() + 1
        auc = (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    else:
        auc = 0.5

    return {
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
        "roc_auc":   round(float(auc), 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ============================================================
# Training loop
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in tqdm(loader, desc="  train", leave=False, unit="batch"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []
    for x, y in tqdm(loader, desc="  val", leave=False, unit="batch"):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * x.size(0)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    metrics = binary_metrics(all_logits, all_labels)
    return total_loss / len(loader.dataset), metrics


def main():
    parser = argparse.ArgumentParser(description="Train temporal LSTM on FF++ embeddings")
    parser.add_argument("--embeddings-root", required=True,
                        help="Path to embeddings/ folder (with real/ and fake/ subdirs)")
    parser.add_argument("--out-dir", default="training/models",
                        help="Where to save the trained model + metrics")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_ds = FFEmbeddingsDataset(
        args.embeddings_root, seq_len=args.seq_len, split="train", seed=args.seed
    )
    val_ds = FFEmbeddingsDataset(
        args.embeddings_root, seq_len=args.seq_len, split="val", seed=args.seed
    )
    print(f"Train: {len(train_ds)} sequences  |  Val: {len(val_ds)} sequences")
    print(f"Train label dist (real, fake): {train_ds.label_distribution()}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ------------------------------------------------------------------
    # Model + optimizer
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device.upper()}")

    model = TemporalLSTM(embedding_dim=768, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    best_f1 = -1.0
    best_metrics = None
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0

        print(
            f"[epoch {epoch:02d}/{args.epochs}]  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"acc={val_metrics['accuracy']:.3f}  "
            f"f1={val_metrics['f1']:.3f}  "
            f"auc={val_metrics['roc_auc']:.3f}  "
            f"({dt:.1f}s)"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_loss,   4),
            **val_metrics,
            "elapsed_sec": round(dt, 1),
        })

        # Save best-by-F1 checkpoint
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_metrics = val_metrics
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "embedding_dim": 768,
                    "hidden_dim": args.hidden_dim,
                    "seq_len": args.seq_len,
                },
                "epoch": epoch,
                "val_metrics": val_metrics,
            }, out_dir / "lstm_temporal.pt")

    # ------------------------------------------------------------------
    # Save final metrics
    # ------------------------------------------------------------------
    with open(out_dir / "lstm_metrics.json", "w") as f:
        json.dump({
            "best_val_metrics": best_metrics,
            "history": history,
            "config": vars(args),
        }, f, indent=2)

    print("\n" + "=" * 60)
    print(f"BEST VAL METRICS")
    for k, v in best_metrics.items():
        print(f"  {k:12s}: {v}")
    print(f"\nSaved to: {out_dir / 'lstm_temporal.pt'}")
    print(f"Metrics : {out_dir / 'lstm_metrics.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()