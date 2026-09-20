# SATYA NETRA — LSTM Temporal Model Training

This folder contains everything needed to train the custom LSTM temporal
classifier on the FaceForensics++ dataset. The main app (`app.py`) uses
pre-trained ViT + EfficientNet + Audio models via HuggingFace, but this
folder is where **we train our own model** on real deepfake data.

**Runs on GPU.** CPU is technically possible but ~10x slower.
Recommended: RTX 3060 / 4050 or better, 8+ GB VRAM.

---

## Pipeline overview
FF++ videos (.mp4)
↓ extract_faces.py
face crops (.jpg)
↓ generate_embeddings.py
ViT embeddings (.npy, one per video)
↓ train_lstm.py
lstm_temporal.pt ← the trained model, committed back to repo

---

## Step 0 — Environment setup

From `backend/` with the venv already active:

```bash
pip install -r requirements.txt
```

Verify GPU is detected:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

Should print `CUDA: True NVIDIA GeForce RTX 4050 Laptop GPU` (or similar).
If it prints `CUDA: False`, install the CUDA build of torch:

```bash
pip uninstall torch torchvision -y
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

*(Use `cu124` instead of `cu121` if your NVIDIA driver supports CUDA 12.4+;
check with `nvidia-smi`.)*

---

## Step 1 — Get the FaceForensics++ dataset

1. Go to <https://github.com/ondyari/FaceForensics>
2. Fill out the access request form (Google Form linked in the README).
   Approval takes **24–48 hours** — start this early.
3. When approved, you get an email with a download script (`faceforensics_download_v4.py`).
4. Download the **c23 (medium quality) subset** — we don't need the massive raw one:

```bash
# Download REAL videos (~1000 videos, ~5 GB)
python faceforensics_download_v4.py training/data -d original -c c23 -t videos

# Download FAKE videos (one command per method, ~1000 videos each, ~5 GB each)
python faceforensics_download_v4.py training/data -d Deepfakes      -c c23 -t videos
python faceforensics_download_v4.py training/data -d Face2Face      -c c23 -t videos
python faceforensics_download_v4.py training/data -d FaceSwap       -c c23 -t videos
python faceforensics_download_v4.py training/data -d NeuralTextures -c c23 -t videos
```

**Storage needed:** ~25 GB for all methods, or ~10 GB for just `Deepfakes` if
you want to start small.

**Time:** 1-3 hours depending on internet speed.

Verify layout:
training/data/
├── original_sequences/youtube/c23/videos/.mp4
└── manipulated_sequences/
├── Deepfakes/c23/videos/.mp4
├── Face2Face/c23/videos/.mp4
├── FaceSwap/c23/videos/.mp4
└── NeuralTextures/c23/videos/*.mp4

---

## Step 2 — Extract face crops (~20-30 min on RTX 4050)

**Quick test first** (5 videos per category, ~2 minutes):

```bash
python training/extract_faces.py \
    --data-root   training/data \
    --output-root training/faces \
    --max-videos  5
```

Check `training/faces/real/` and `training/faces/fake/` — you should see
subfolders each containing ~20-50 .jpg face crops. If yes, do the full run:

```bash
python training/extract_faces.py \
    --data-root   training/data \
    --output-root training/faces
```

Safe to interrupt and re-run — it skips videos already processed.

---

## Step 3 — Generate ViT embeddings (~15-25 min on RTX 4050)

First run downloads the ViT model (~350 MB, one-time).

```bash
python training/generate_embeddings.py \
    --faces-root      training/faces \
    --embeddings-root training/embeddings
```

Output: one `.npy` file per video in `training/embeddings/real/` and
`training/embeddings/fake/`. Each `.npy` is a `(T, 768)` array — T frames
of 768-dim CLS-token embeddings.

Safe to interrupt and re-run.

---

## Step 4 — Train the LSTM (~3-5 min on RTX 4050)

```bash
python training/train_lstm.py \
    --embeddings-root training/embeddings
```

**Output files** (both need to be committed to the repo):

- `training/models/lstm_temporal.pt`   — trained weights (~5 MB)
- `training/models/lstm_metrics.json`  — accuracy/precision/recall/F1/AUC

**Expected metrics on full FF++ subset:**
- Val accuracy: 85-92%
- ROC AUC:      0.90-0.96
- (Actual numbers depend on the subset trained on; report what you get.)

---

## Step 5 — Commit the trained model back to the repo

```bash
cd ..           # back to project root
git add backend/training/models/lstm_temporal.pt backend/training/models/lstm_metrics.json
git commit -m "Add trained LSTM temporal model + metrics"
git push
```

**Do NOT commit:**
- `training/data/`         (25 GB of videos — never)
- `training/faces/`        (large intermediate output)
- `training/embeddings/`   (large intermediate output)

These are already covered by adding this to `.gitignore` (see below).

---

## `.gitignore` additions

Make sure the project's `.gitignore` includes:
backend/training/data/
backend/training/faces/
backend/training/embeddings/
!backend/training/models/ # DO commit trained model weights

---

## Configuration knobs

Passed as CLI flags to `train_lstm.py`:

| Flag              | Default | What it does |
|-------------------|---------|--------------|
| `--epochs`        | 20      | More = better fit, but risks overfitting past ~30 |
| `--batch-size`    | 32      | Reduce to 16 if GPU OOM |
| `--lr`            | 1e-3    | Learning rate — leave alone unless loss plateaus |
| `--seq-len`       | 30      | Frames per sequence (~1 sec of video at 30fps) |
| `--hidden-dim`    | 128     | LSTM capacity — larger = more parameters |

---

## Once trained — how it plugs into the live app

The main app (`app.py`) will import a new module `detectors/temporal.py`
(created by the main dev team) that loads `lstm_temporal.pt` and feeds
its output into the existing multimodal fusion engine as a fourth signal
alongside video_primary, video_secondary, and audio.

Your job as the trainer ends at Step 5. The integration happens on the
main app side.

---

## Troubleshooting

**`CUDA out of memory` during embedding generation**
→ Edit `generate_embeddings.py` and reduce `BATCH_SIZE = 32` to `16` or `8`.

**`No .mp4 files in ...` during face extraction**
→ FF++ download didn't complete for that method — re-run the download command.

**Face extraction finds 0 faces in most videos**
→ Some videos in FF++ start with black frames. Increase `FRAME_STRIDE` in
   `extract_faces.py` (from 5 to 3) so more frames are sampled, or reduce
   `MIN_FACES` from 10 to 5.

**Training loss stays flat / accuracy stuck at 50%**
→ Check `train_ds.label_distribution()` output at the start of training.
   If it's very unbalanced (e.g., 100 real vs 4000 fake), the LSTM defaults
   to always predicting the majority class. Fix: use a subset of the fake
   folders (drop 2 of the 4 methods) to rebalance.

**GPU not detected inside training script but works standalone**
→ Wrong torch build installed. Reinstall with CUDA index-url (see Step 0).

---

## Contact Aditya Gahlan

Ping the main dev if you hit anything not covered above.