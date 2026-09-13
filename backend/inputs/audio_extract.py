"""
Audio extraction utilities.

Extracts audio waveform from an uploaded video file using the bundled ffmpeg
binary that ships with imageio-ffmpeg. This lets us process audio without
requiring the user to install ffmpeg separately.
"""
import subprocess
import tempfile
import wave

import numpy as np
import imageio_ffmpeg


def extract_audio_from_video(video_path, target_sample_rate=16000):
    """
    Extract mono audio from a video file at the given sample rate.

    Args:
        video_path: absolute path to a video file (mp4, mov, avi, webm, ...)
        target_sample_rate: desired output sample rate in Hz (default 16000
                            because most audio deepfake models expect 16 kHz)

    Returns:
        (waveform, sample_rate) tuple:
          - waveform: 1D numpy array of float32 samples in [-1.0, 1.0]
          - sample_rate: int (equals target_sample_rate)
        OR (None, None) if extraction fails (e.g., video has no audio track)
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    # Extract to a temp WAV file, then load into numpy
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    cmd = [
        ffmpeg_exe,
        "-y",                          # overwrite output
        "-i", video_path,              # input video
        "-vn",                         # ignore video stream
        "-ac", "1",                    # mono
        "-ar", str(target_sample_rate),
        "-f", "wav",
        "-loglevel", "error",          # suppress verbose output
        wav_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            print(f"[audio_extract] ffmpeg failed: {result.stderr.decode(errors='ignore')}")
            return None, None
    except subprocess.TimeoutExpired:
        print("[audio_extract] ffmpeg timed out")
        return None, None
    except Exception as e:
        print(f"[audio_extract] ffmpeg error: {e}")
        return None, None

    # Load the WAV into a numpy array
    try:
        with wave.open(wav_path, "rb") as wf:
            n_frames = wf.getnframes()
            sample_width = wf.getsampwidth()
            raw = wf.readframes(n_frames)

        # Convert bytes to numpy based on sample width
        if sample_width == 2:
            dtype = np.int16
            max_val = 32768.0
        elif sample_width == 4:
            dtype = np.int32
            max_val = 2147483648.0
        else:
            dtype = np.uint8
            max_val = 128.0

        audio = np.frombuffer(raw, dtype=dtype).astype(np.float32) / max_val
        return audio, target_sample_rate
    except Exception as e:
        print(f"[audio_extract] WAV load failed: {e}")
        return None, None


def slice_audio_for_frame(full_audio, sample_rate, frame_index, video_fps, window_seconds=1.0):
    """
    Grab an audio slice centered on a specific video frame.

    Useful for aligning audio inference with video-frame processing:
    when we process video frame N, we analyze the audio around that timestamp.

    Args:
        full_audio: complete audio waveform from extract_audio_from_video
        sample_rate: audio sample rate
        frame_index: which video frame we're currently processing
        video_fps: source video's frame rate
        window_seconds: how much audio context to grab (centered on the frame)

    Returns:
        1D numpy audio slice, or None if the slice would be out of bounds
    """
    if full_audio is None or len(full_audio) == 0:
        return None

    frame_time_sec = frame_index / max(video_fps, 1.0)
    center_sample = int(frame_time_sec * sample_rate)
    half_window = int((window_seconds / 2) * sample_rate)

    start = max(0, center_sample - half_window)
    end = min(len(full_audio), center_sample + half_window)

    if end - start < sample_rate * 0.1:   # less than 100 ms → skip
        return None

    return full_audio[start:end]