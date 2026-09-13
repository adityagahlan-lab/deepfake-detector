"""
Audio deepfake detection module.

STATUS: STUB (Phase 2 implementation pending).

Phase 2 will load a pre-trained audio deepfake model (e.g., MelodyMachine/
Deepfake-audio-detection-V2) and process raw audio waveforms to detect
synthesized/cloned voices.

For now, this returns None so the fusion layer knows to skip audio-based scoring.
"""
import config


class AudioDeepfakeDetector:
    """Placeholder — will wrap a HuggingFace audio classification model in Phase 2."""

    def __init__(self, model_name=None):
        self.model_name = model_name or config.AUDIO_MODEL
        self.model = None  # will be loaded in Phase 2

    def load(self):
        """No-op for now. Phase 2 will download and load the audio model here."""
        pass

    def predict(self, audio_waveform, sample_rate=16000):
        """
        Return fake-confidence score for an audio segment.

        Args:
            audio_waveform: 1D numpy array of audio samples
            sample_rate: audio sampling rate in Hz

        Returns:
            float in [0.0, 1.0] — higher = more likely fake voice
            OR None if audio detection is not yet implemented
        """
        # Phase 2: implement using self.model
        return None

    def is_available(self):
        """Returns True once the real audio model is loaded and ready."""
        return self.model is not None
    