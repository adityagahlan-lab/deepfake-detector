"""
WebRTC / live-stream input source.

STATUS: STUB (Phase 3 implementation pending).

Phase 3 will accept browser-originating WebRTC streams (so remote users
can point their camera at the platform without any local install) and
RTSP streams from IP cameras / OBS setups.

Rough implementation sketch for later:
  - Use aiortc for WebRTC signaling + media handling
  - Use OpenCV's VideoCapture with an RTSP URL for RTSP mode
  - Yield frames in the same shape as WebcamSource / VideoFileSource
    so the rest of the pipeline stays unchanged.
"""


class WebRTCSource:
    """Placeholder — Phase 3 will implement browser-originated WebRTC streams."""

    def __init__(self, signaling_url=None):
        self.signaling_url = signaling_url
        self._connected = False

    def open(self):
        raise NotImplementedError(
            "WebRTC input is not yet implemented. Planned for Phase 3."
        )

    def read(self):
        raise NotImplementedError

    def release(self):
        pass


class RTSPSource:
    """Placeholder — Phase 3 will implement RTSP stream input (IP cameras / OBS)."""

    def __init__(self, rtsp_url=None):
        self.rtsp_url = rtsp_url

    def open(self):
        raise NotImplementedError(
            "RTSP input is not yet implemented. Planned for Phase 3."
        )

    def read(self):
        raise NotImplementedError

    def release(self):
        pass
    