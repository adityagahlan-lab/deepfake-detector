"""
Live microphone audio capture.

Runs a background thread that continuously fills a ring buffer with
the latest N seconds of audio. Any consumer can grab the "last 1 second"
on demand without blocking or coordinating with the audio callback.

Why a ring buffer:
  - Audio arrives at 16 kHz continuously in small chunks
  - The video pipeline runs at ~1-3 Hz (much slower)
  - We just want "give me the last second of audio right now"
  - A ring buffer decouples the two rates cleanly
"""
import threading
import numpy as np

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):
    # OSError = no audio backend on the machine (rare but possible)
    SOUNDDEVICE_AVAILABLE = False


class MicrophoneSource:
    """
    Continuously captures mic audio into a ring buffer.

    Usage:
        mic = MicrophoneSource()
        mic.start()
        ...
        recent_audio = mic.get_last_seconds(1.0)   # numpy array, 1 sec @ 16kHz
        ...
        mic.stop()
    """

    def __init__(self, sample_rate=16000, buffer_seconds=3.0, device=None):
        self.sample_rate = sample_rate
        self.buffer_seconds = buffer_seconds
        self.buffer_size = int(sample_rate * buffer_seconds)
        self.device = device                   # None = system default input
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_pos = 0                     # next slot in the ring
        self.lock = threading.Lock()
        self.stream = None
        self._running = False
        self._error = None

    def _callback(self, indata, frames, time_info, status):
        """Called by sounddevice from its audio thread whenever new samples arrive."""
        if status:
            # status is non-empty on overflows/underflows; not fatal, log only
            print(f"[MicrophoneSource] status: {status}")

        # indata shape: (frames, channels). Take mono.
        mono = indata[:, 0] if indata.ndim > 1 else indata

        with self.lock:
            end = self.write_pos + len(mono)
            if end <= self.buffer_size:
                self.buffer[self.write_pos:end] = mono
            else:
                # Wrap around
                first_part = self.buffer_size - self.write_pos
                self.buffer[self.write_pos:] = mono[:first_part]
                self.buffer[:len(mono) - first_part] = mono[first_part:]
            self.write_pos = (self.write_pos + len(mono)) % self.buffer_size

    def start(self):
        """Open the mic and start capturing. Returns True on success."""
        if not SOUNDDEVICE_AVAILABLE:
            self._error = "sounddevice library not installed or no audio backend"
            return False

        if self._running:
            return True

        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=0,               # let sounddevice pick optimal
                device=self.device,
                callback=self._callback,
            )
            self.stream.start()
            self._running = True
            self._error = None
            return True
        except Exception as e:
            self._error = str(e)
            self._running = False
            return False

    def stop(self):
        """Stop capturing and release the mic."""
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self._running = False

    def get_last_seconds(self, seconds=1.0):
        """
        Return the most recent `seconds` of audio as a 1D numpy array.
        Returns None if the mic isn't running.
        """
        if not self._running:
            return None

        n_samples = min(int(seconds * self.sample_rate), self.buffer_size)

        with self.lock:
            # Read backwards from write_pos, wrapping if needed
            end = self.write_pos
            start = (end - n_samples) % self.buffer_size

            if start < end:
                slice_out = self.buffer[start:end].copy()
            else:
                # Wraps around the end of the ring
                slice_out = np.concatenate([
                    self.buffer[start:],
                    self.buffer[:end],
                ])

        return slice_out

    def get_current_level(self, seconds=0.1):
        
        if not self._running:
            return 0.0

        n_samples = min(int(seconds * self.sample_rate), self.buffer_size)
        with self.lock:
            end = self.write_pos
            start = (end - n_samples) % self.buffer_size
            if start < end:
                recent = self.buffer[start:end]
            else:
                recent = np.concatenate([self.buffer[start:], self.buffer[:end]])

        if len(recent) == 0:
            return 0.0

        # RMS amplitude, scaled and clipped to [0, 1]
        rms = float(np.sqrt(np.mean(recent ** 2)))
        # Boost quiet signals so bar is visible during normal speech
        level = min(1.0, rms * 5.0)
        return level

    def is_running(self):
        return self._running

    def last_error(self):
        return self._error