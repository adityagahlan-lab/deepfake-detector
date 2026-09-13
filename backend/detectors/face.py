"""
Face detection module.
Currently uses OpenCV's Haar Cascade (fast, CPU-friendly).
Future: swap to MediaPipe for better accuracy on side profiles / low light.
"""
import cv2
import config


class FaceDetector:
    """Detects faces in a BGR frame and returns bounding boxes."""

    def __init__(self):
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect(self, frame_bgr):
        """
        Detect faces in a BGR image.

        Args:
            frame_bgr: numpy array (H, W, 3) in BGR format from OpenCV

        Returns:
            List of (x, y, w, h) tuples, one per detected face
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=config.FACE_SCALE_FACTOR,
            minNeighbors=config.FACE_MIN_NEIGHBORS,
            minSize=(config.FACE_MIN_SIZE, config.FACE_MIN_SIZE),
        )
        return list(faces)

    def crop_face(self, frame_bgr, box):
        """
        Extract a face region from a frame.

        Args:
            frame_bgr: full frame
            box: (x, y, w, h) tuple from detect()

        Returns:
            Cropped face as numpy array, or None if invalid
        """
        x, y, w, h = box
        crop = frame_bgr[y:y + h, x:x + w]
        return crop if crop.size > 0 else None