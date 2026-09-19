"""
LSTM Temporal Deepfake Classifier.

Input:  sequence of frame embeddings (T frames × D-dim vectors)
        where D is the ViT embedding size (typically 768).
Output: single fake-confidence score in [0, 1] for the whole sequence.

The idea: per-frame classifiers see spatial artefacts (blurred edges,
odd textures). This model sees TEMPORAL inconsistencies — flicker,
warping, unnatural motion — that only appear when you look at frames
as a sequence.

Small on purpose:
  - Hidden size 128 is enough for binary classification
  - 2 layers is a reasonable depth without overfitting on limited data
  - Dropout in the LSTM + before the head prevents memorisation
"""
import torch
import torch.nn as nn


class TemporalLSTM(nn.Module):
    def __init__(self,
                 embedding_dim: int = 768,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 bidirectional: bool = True):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # If bidirectional, the LSTM output size doubles
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        # Classification head: LSTM output -> single logit
        self.head = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),      # single logit (we apply sigmoid outside for prob)
        )

    def forward(self, x, lengths=None):
        """
        Args:
            x: tensor of shape (batch_size, seq_len, embedding_dim)
            lengths: optional tensor of true sequence lengths per batch item
                     (for handling variable-length sequences via packing)

        Returns:
            logits: tensor of shape (batch_size,) — raw scores before sigmoid
        """
        # Run the LSTM over the sequence
        # We ignore packing for simplicity — for a first version, batch items
        # are padded to the same length.
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take the FINAL time-step's output as the sequence summary
        # (bidirectional: this concatenates last forward + last backward automatically
        #  because we're using the output tensor, not h_n)
        final_output = lstm_out[:, -1, :]         # (batch, lstm_out_dim)

        logits = self.head(final_output).squeeze(-1)   # (batch,)
        return logits

    def predict_proba(self, x, lengths=None):
        """
        Returns fake-confidence probabilities in [0, 1].
        Wraps forward() with sigmoid — this is what the live inference
        pipeline will call.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, lengths)
            return torch.sigmoid(logits)


# ============================================================
# Quick sanity check when run standalone: `python lstm_model.py`
# ============================================================
if __name__ == "__main__":
    model = TemporalLSTM()
    print(model)

    # Fake batch: 4 sequences, each 30 frames, each with a 768-dim embedding
    dummy = torch.randn(4, 30, 768)
    logits = model(dummy)
    probs = torch.sigmoid(logits)

    print(f"\nInput shape:  {dummy.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Probs:        {probs}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")