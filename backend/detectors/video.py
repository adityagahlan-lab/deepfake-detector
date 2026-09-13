"""
Video deepfake detection module.
Loads a pre-trained Vision Transformer from HuggingFace and returns a fake-confidence
score for a cropped face image.

Phase 2 will add a second model here for ensemble voting.
"""
import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

import config


class VideoDeepfakeDetector:
    """Wraps a HuggingFace image classification model as a deepfake detector."""

    def __init__(self, model_name=None):
        self.model_name = model_name or config.VIDEO_MODEL_PRIMARY
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self._fake_idx = None

    def load(self):
        """Download (if needed) and load the model. Cached after first call."""
        if self.model is not None:
            return  # already loaded

        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        # Find which output index corresponds to "fake"
        labels = self.model.config.id2label
        self._fake_idx = next(
            i for i, name in labels.items() if "fake" in name.lower()
        )

    def predict(self, face_bgr):
        """
        Return fake-confidence score (0.0 to 1.0) for a face crop.

        Args:
            face_bgr: cropped face as numpy array in BGR (from OpenCV)

        Returns:
            float in [0.0, 1.0] — higher = more likely deepfake
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        # Resize + convert BGR to RGB (PIL/transformers expect RGB)
        face_bgr = cv2.resize(face_bgr, (config.FACE_INPUT_SIZE, config.FACE_INPUT_SIZE))
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(face_rgb)

        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

        return probs[self._fake_idx].item()