"""
Multimodal Fusion Engine — SPARTA's core novelty layer.

Takes independent evidence signals from multiple detectors and produces
a single, calibrated deepfake risk score.

Current signals:
  - video_primary   : ViT-based visual deepfake detector (implemented)
  - video_secondary : EfficientNet-based visual detector (Phase 2)
  - audio           : audio deepfake detector          (Phase 2)
  - lipsync         : audio-visual sync verification   (Phase 2)
  - rppg            : blood-flow signal from face      (Phase 2 stretch)

Fusion strategy:
  Weighted average of available signals, with automatic re-normalisation
  when some signals are unavailable (e.g., no audio track, or model not loaded).
  Weights come from config.py so they can be tuned without code changes.

Why this is novel:
  No off-the-shelf tool combines exactly these signals with this weighting +
  temporal aggregation. The individual detectors are pre-trained;
  the *integration* is our contribution.
"""
import config


class MultimodalFusion:
    """
    Combines evidence from multiple deepfake detectors into a single risk score.
    Missing signals (None values) are ignored and weights re-normalised.
    """

    def __init__(self,
                 w_video_primary=None,
                 w_video_secondary=None,
                 w_audio=None):
        self.weights = {
            "video_primary": w_video_primary if w_video_primary is not None else config.FUSION_WEIGHT_VIDEO_PRIMARY,
            "video_secondary": w_video_secondary if w_video_secondary is not None else config.FUSION_WEIGHT_VIDEO_SECONDARY,
            "audio": w_audio if w_audio is not None else config.FUSION_WEIGHT_AUDIO,
        }

    def fuse(self, scores: dict):
        """
        Combine per-detector scores into one risk score.

        Args:
            scores: dict with keys in {"video_primary", "video_secondary",
                    "audio", ...} and values in [0.0, 1.0] or None.
                    None means "signal unavailable" and gets skipped.

        Returns:
            dict with keys:
              - fused_score       : final risk score in [0.0, 1.0]
              - contributing      : list of signal names that were used
              - per_signal        : {name: score} of the signals that contributed
              - effective_weights : {name: normalised_weight} used in this fusion
        """
        # Filter to signals that were actually provided
        available = {k: v for k, v in scores.items() if v is not None and k in self.weights}

        if not available:
            return {
                "fused_score": 0.0,
                "contributing": [],
                "per_signal": {},
                "effective_weights": {},
            }

        # Re-normalise weights across the signals we actually have
        raw_weights = {k: self.weights[k] for k in available}
        total_w = sum(raw_weights.values())
        norm_weights = {k: w / total_w for k, w in raw_weights.items()}

        fused = sum(available[k] * norm_weights[k] for k in available)

        return {
            "fused_score": fused,
            "contributing": list(available.keys()),
            "per_signal": available,
            "effective_weights": norm_weights,
        }

    def explanation(self, fusion_result):
        """
        Human-readable one-liner explaining why the fused score is what it is.
        Useful for the dashboard's 'Why?' tooltip and for judge Q&A.
        """
        if not fusion_result["contributing"]:
            return "No signals available for fusion."

        parts = []
        for name in fusion_result["contributing"]:
            score = fusion_result["per_signal"][name]
            weight = fusion_result["effective_weights"][name]
            parts.append(f"{name}={score * 100:.0f}% (w={weight:.2f})")
        return " + ".join(parts) + f"  →  {fusion_result['fused_score'] * 100:.1f}%"