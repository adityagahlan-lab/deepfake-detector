"""
Central configuration for SATYA NETRA Deepfake Detection Platform.
All tunable parameters live here so teammates can adjust without hunting.
"""

# ============================================================
# TEMPORAL AGGREGATION
# ============================================================
WINDOW_SIZE = 5                      # Rolling window: average last N model scores
ALERT_THRESHOLD = 0.6                # Smoothed score above this = alert territory
CONSECUTIVE_ALERTS_NEEDED = 3        # Consecutive over-threshold checks before firing an alert

# ============================================================
# PERFORMANCE
# ============================================================
FRAME_SKIP = 10                      # Run model every Nth frame (set to 1 on GPU)
WEBCAM_WIDTH = 480                   # Webcam capture resolution (set to 1280 on GPU)
WEBCAM_HEIGHT = 360                  # Webcam capture resolution (set to 720 on GPU)
FACE_INPUT_SIZE = 224                # Face crops resized to this before model inference

# ============================================================
# FACE DETECTION
# ============================================================
FACE_MIN_SIZE = 60                   # Minimum face size (pixels) to detect
FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGHBORS = 5

# ============================================================
# CLASSIFICATION THRESHOLDS
# ============================================================
REAL_THRESHOLD = 0.4                 # Below this = REAL
SUSPICIOUS_THRESHOLD = 0.7           # Below this = SUSPICIOUS, above = FAKE

# ============================================================
# MODELS
# ============================================================
# Video deepfake models (pre-trained, downloaded from HuggingFace on first run)
VIDEO_MODEL_PRIMARY = "prithivMLmods/Deep-Fake-Detector-Model"
VIDEO_MODEL_SECONDARY = "dima806/deepfake_vs_real_image_detection"  # for future ensemble

# Audio deepfake model (for Phase 2)
AUDIO_MODEL = "MelodyMachine/Deepfake-audio-detection-V2"

# ============================================================
# FUSION WEIGHTS (multi-model ensemble)
# ============================================================
FUSION_WEIGHT_VIDEO_PRIMARY = 0.35
FUSION_WEIGHT_VIDEO_SECONDARY = 0.25
FUSION_WEIGHT_AUDIO = 0.15
FUSION_WEIGHT_TEMPORAL = 0.15         # our custom-trained LSTM

# ============================================================
# LSTM TEMPORAL MODEL
# ============================================================
TEMPORAL_SEQUENCE_LEN = 30            # frames buffered before temporal inference
TEMPORAL_MIN_LEN = 30                 # minimum frames before we even try