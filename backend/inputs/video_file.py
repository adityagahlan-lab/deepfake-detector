"""
Video-file input source.

Reads frames from an uploaded video file, with two playback modes:
  - real_time  : skip frames to match the source video's FPS
                 (useful when the inference pipeline can't keep up)
  - full       : read every frame (slow but analyses everything)
"""
import cv2


class VideoFileSource:
    """
    Frame source that reads from a video file on disk.

    Usage:
        src = VideoFileSource("/path/to/video.mp4", playback_mode="real_time",
                              pipeline_fps=2)
        src.open()
        for frame in src.frames():
            ...
        src.release()
    """

    def __init__(self, path, playback_mode="real_time", pipeline_fps=2):
        """
        Args:
            path: absolute path to the video file
            playback_mode: "real_time" (skip to keep up) or "full" (every frame)
            pipeline_fps: expected downstream processing FPS (only used in real_time mode)
        """
        self.path = path
        self.playback_mode = playback_mode
        self.pipeline_fps = pipeline_fps
        self.cap = None
        self.video_fps = 30.0
        self.total_frames = 0
        self.read_skip = 1
        self._processed_count = 0

    def open(self):
        """Open the video file and configure frame skipping."""
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file: {self.path}")

        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._processed_count = 0

        if self.playback_mode == "real_time":
            self.read_skip = max(1, int(self.video_fps / self.pipeline_fps))
        else:
            self.read_skip = 1

    def read(self):
        """
        Read next frame (respecting playback_mode's skip setting).

        Returns:
            (success: bool, frame: numpy array or None)
        """
        if self.cap is None:
            raise RuntimeError("Video file not opened. Call .open() first.")

        # Skip N-1 frames without decoding (fast), then read the Nth
        for _ in range(self.read_skip - 1):
            self.cap.grab()
            self._processed_count += 1

        ret, frame = self.cap.read()
        self._processed_count += 1
        return ret, frame

    def progress(self):
        """Fraction of the video processed so far, in [0.0, 1.0]."""
        if self.total_frames <= 0:
            return 0.0
        return min(self._processed_count / self.total_frames, 1.0)

    def info(self):
        """Metadata dict for display in the UI."""
        return {
            "video_fps": self.video_fps,
            "total_frames": self.total_frames,
            "read_skip": self.read_skip,
            "playback_mode": self.playback_mode,
        }

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        