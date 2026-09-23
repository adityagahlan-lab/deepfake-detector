"""
Screen capture input source.
Analyses any live stream playing on screen: Instagram Live, YouTube Live,
Zoom, Google Meet, news broadcasts.
"""
import cv2
import numpy as np

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False


class ScreenSource:
    def __init__(self, monitor_index=1, region=None, max_width=960):
        self.monitor_index = monitor_index
        self.region = region          # dict(left, top, width, height) or None = full monitor
        self.max_width = max_width
        self.sct = None
        self.bbox = None

    def open(self):
        if not MSS_AVAILABLE:
            raise RuntimeError("mss not installed. Run: pip install mss")
        self.sct = mss.mss()
        monitors = self.sct.monitors
        if self.monitor_index >= len(monitors):
            raise RuntimeError(f"Monitor {self.monitor_index} not found.")
        mon = monitors[self.monitor_index]
        if self.region:
            self.bbox = {
                "left": mon["left"] + self.region["left"],
                "top": mon["top"] + self.region["top"],
                "width": self.region["width"],
                "height": self.region["height"],
            }
        else:
            self.bbox = dict(mon)

    def read(self):
        if self.sct is None:
            raise RuntimeError("Screen source not opened.")
        shot = self.sct.grab(self.bbox)
        frame = np.ascontiguousarray(np.array(shot)[:, :, :3])   # BGRA -> BGR
        h, w = frame.shape[:2]
        if w > self.max_width:
            frame = cv2.resize(frame, (self.max_width, int(h * self.max_width / w)))
        return True, frame

    def release(self):
        if self.sct is not None:
            self.sct.close()
            self.sct = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.release()

    @staticmethod
    def list_monitors():
        if not MSS_AVAILABLE:
            return []
        with mss.mss() as s:
            return s.monitors[1:]