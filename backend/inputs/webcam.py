"""
Webcam input source.
Wraps OpenCV's VideoCapture with SPARTA's configured resolution.
"""
import cv2

import config


class WebcamSource:
    """
    Live webcam capture using OpenCV.

    Usage:
        with WebcamSource() as cam:
            for frame in cam.frames():
                ...
    """

    def __init__(self, device_index=0, mirror=True):
        self.device_index = device_index
        self.mirror = mirror
        self.cap = None

    def open(self):
        """Open the webcam. Raises RuntimeError if it fails."""
        self.cap = cv2.VideoCapture(self.device_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.WEBCAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WEBCAM_HEIGHT)

        if not self.cap.isOpened():
            raise RuntimeError(
                "Could not open webcam. Is another app using it, "
                "or is the wrong device index set?"
            )

    def read(self):
        """
        Read one frame.

        Returns:
            (success: bool, frame: numpy array or None)
        """
        if self.cap is None:
            raise RuntimeError("Webcam not opened. Call .open() first.")

        ret, frame = self.cap.read()
        if not ret:
            return False, None

        if self.mirror:
            frame = cv2.flip(frame, 1)
        return True, frame

    def release(self):
        """Release the webcam handle."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # Context-manager support (so we can use `with WebcamSource() as cam:`)
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        