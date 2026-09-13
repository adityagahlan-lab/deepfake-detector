"""
Audio deepfake detection module.

Loads a pre-trained audio classification model from HuggingFace and returns
a fake-confidence score for an audio waveform. Detects synthesized voices,
voice cloning, and TTS-generated speech.
"""
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

import config


class AudioDeepfakeDetector:
    """Wraps a HuggingFace audio classification model as a deepfake detector."""

    def __init__(self, model_name=None):
        self.model_name = model_name or config.AUDIO_MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self._fake_idx = None
        self._loaded = False

    def load(self):
        """Download (if needed) and load the model. Cached after first call."""
        if self._loaded:
            return

        try:
            self.processor = AutoFeatureExtractor.from_pretrained(self.model_name)
            self.model = AutoModelForAudioClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

            # Find which output index corresponds to "fake" / "spoof" / "synthetic"
            labels = self.model.config.id2label
            self._fake_idx = self._find_fake_index(labels)
            self._loaded = True
            print(f"[AudioDetector] Loaded {self.model_name} on {self.device}")
        except Exception as e:
            print(f"[AudioDetector] Failed to load model: {e}")
            self._loaded = False

    def _find_fake_index(self, labels: dict) -> int:
        """
        Different audio deepfake models use different label names:
        'fake', 'spoof', 'synthetic', 'ai', 'bonafide' (which is real), etc.
        Try common patterns.
        """
        fake_keywords = ["fake", "spoof", "synthetic", "ai", "generated", "deepfake"]
        real_keywords = ["real", "bonafide", "genuine", "human", "original"]

        for idx, name in labels.items():
            name_lower = name.lower()
            if any(kw in name_lower for kw in fake_keywords):
                return idx
        # If none matched, invert: find the "real" label and use the OTHER index
        for idx, name in labels.items():
            if any(kw in name.lower() for kw in real_keywords):
                # Return the other index (assumes binary classifier)
                return 1 - idx if len(labels) == 2 else idx
        # Fallback: assume index 1 is "fake" (binary classifier convention)
        return 1

    def predict(self, audio_waveform, sample_rate=16000):
        """
        Return fake-confidence score for an audio segment.

        Args:
            audio_waveform: 1D numpy array of audio samples (mono, float32)
            sample_rate: audio sampling rate in Hz. Will be resampled to model's rate.

        Returns:
            float in [0.0, 1.0] — higher = more likely fake voice
            OR None if audio detection isn't available or input is invalid
        """
        if not self._loaded or self.model is None:
            return None

        if audio_waveform is None or len(audio_waveform) == 0:
            return None

        # Ensure mono float32
        if audio_waveform.ndim > 1:
            audio_waveform = audio_waveform.mean(axis=0)
        audio_waveform = audio_waveform.astype(np.float32)

        try:
            inputs = self.processor(
                audio_waveform,
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

            return float(probs[self._fake_idx].item())
        except Exception as e:
            print(f"[AudioDetector] Inference error: {e}")
            return None

    def is_available(self):
        """True if the model is loaded and ready to predict."""
        return self._loaded