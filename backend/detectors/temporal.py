"""
Temporal deepfake detection module.

Loads the LSTM temporal classifier that WE trained on FaceForensics++
using ViT embeddings, and provides fake-confidence scores for sequences
of frame embeddings.

Unlike the other detectors, this one operates on a SEQUENCE of frames,
not a single frame. That's what makes it 'temporal' — it can spot
inconsistencies across time (flicker, warping, unnatural motion)
that per-frame classifiers miss.

Fails gracefully:
  - If lstm_temporal.pt is missing → predict() returns None
  - If input sequence is too short → predict() returns None
  - Missing signals are handled by the fusion engine automatically

Usage in app.py (added later, when integrating):
    from detectors.temporal import TemporalLSTMDetector
    from collections import deque

    temporal = TemporalLSTMDetector()
    temporal.load()

    # Somewhere in the frame loop, once we have a ViT embedding for the current face:
    embedding_buffer.append(current_embedding)   # rolling deque of last 30
    if len(embedding_buffer) >= 30:
        temporal_score = temporal.predict(list(embedding_buffer))
"""
from pathlib import Path

import numpy as np
import torch

# The trained model file lives in backend/training/models/
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "training" / "models" / "lstm_temporal.pt"


class TemporalLSTMDetector:
    """Wraps the trained TemporalLSTM as a fusion-compatible detector."""

    def __init__(self, model_path=None, min_sequence_len=15):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.min_sequence_len = min_sequence_len
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.seq_len = None            # sequence length the model was trained with
        self._loaded = False

    def load(self):
        """Load the trained LSTM. No-op (with warning) if the .pt is missing."""
        if self._loaded:
            return

        if not self.model_path.exists():
            print(
                f"[TemporalDetector] No trained model at {self.model_path}. "
                f"Skipping — will return None from predict()."
            )
            self._loaded = False
            return

        try:
            # Import here so app.py doesn't need training/ on the path
            import sys
            training_dir = Path(__file__).resolve().parent.parent / "training"
            sys.path.insert(0, str(training_dir))
            from lstm_model import TemporalLSTM

            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

            cfg = checkpoint.get("config", {})
            self.seq_len = cfg.get("seq_len", 30)

            self.model = TemporalLSTM(
                embedding_dim=cfg.get("embedding_dim", 768),
                hidden_dim=cfg.get("hidden_dim", 128),
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            self._loaded = True

            metrics = checkpoint.get("val_metrics", {})
            print(
                f"[TemporalDetector] Loaded lstm_temporal.pt on {self.device.upper()} "
                f"(val_f1={metrics.get('f1', '?')}, auc={metrics.get('roc_auc', '?')})"
            )
        except Exception as e:
            print(f"[TemporalDetector] Failed to load model: {e}")
            self._loaded = False

    def predict(self, embedding_sequence):
        """
        Return fake-confidence score for a sequence of frame embeddings.

        Args:
            embedding_sequence: list/array of per-frame ViT embeddings,
                                shape (T, D) where D is typically 768.

        Returns:
            float in [0.0, 1.0] — higher = more likely deepfake
            OR None if the detector isn't loaded, or the sequence is too short.
        """
        if not self._loaded or self.model is None:
            return None

        if embedding_sequence is None:
            return None

        # Convert to numpy array if it isn't already
        seq = np.asarray(embedding_sequence, dtype=np.float32)
        if seq.ndim != 2 or seq.shape[0] < self.min_sequence_len:
            return None

        # Match training sequence length: crop or pad
        target = self.seq_len or 30
        T = seq.shape[0]
        if T >= target:
            # Centre crop
            start = (T - target) // 2
            seq = seq[start:start + target]
        else:
            # Zero-pad at the end
            pad = np.zeros((target - T, seq.shape[1]), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=0)

        # Add batch dim → (1, T, D)
        x = torch.from_numpy(seq).unsqueeze(0).to(self.device)

        try:
            with torch.no_grad():
                logits = self.model(x)
                prob = torch.sigmoid(logits)[0].item()
            return float(prob)
        except Exception as e:
            print(f"[TemporalDetector] Inference error: {e}")
            return None

    def is_available(self):
        """True if the trained model is loaded and ready."""
        return self._loaded