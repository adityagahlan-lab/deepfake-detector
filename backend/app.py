"""
SPARTA — Real-Time Deepfake Detection Platform
Streamlit UI layer. All ML / pipeline logic lives in the modules.
"""
import time
import tempfile

import cv2
import streamlit as st

import config
from detectors.face import FaceDetector
from detectors.video import VideoDeepfakeDetector
from detectors.audio import AudioDeepfakeDetector
from pipeline.aggregator import TemporalAggregator
from pipeline.fusion import MultimodalFusion
from inputs.webcam import WebcamSource
from inputs.video_file import VideoFileSource


# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="SPARTA — Deepfake Detection Platform",
    page_icon="🛡",
    layout="wide",
)

st.title("🛡 SPARTA — Real-Time Deepfake Detection Platform")
st.caption("SIH 2026 | USICT028 | AI Safety & Media Verification")


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "Multimodal deepfake detection for live video streams. "
        "Combines multiple pre-trained detectors via a custom fusion engine "
        "with temporal aggregation and streak-based alerting."
    )
    st.markdown("### Pipeline")
    st.markdown(
        "1. Face detection\n"
        "2. Multi-model inference\n"
        "3. Fusion (video + audio)\n"
        "4. Temporal smoothing\n"
        "5. Streak-based alerts"
    )
    st.markdown("### Config")
    st.markdown(f"- Window: `{config.WINDOW_SIZE}`")
    st.markdown(f"- Alert threshold: `{config.ALERT_THRESHOLD}`")
    st.markdown(f"- Streak: `{config.CONSECUTIVE_ALERTS_NEEDED}`")
    st.markdown(f"- Frame skip: `{config.FRAME_SKIP}`")


# ============================================================
# LOAD MODELS (cached across reruns)
# ============================================================
@st.cache_resource
def load_pipeline():
    """Instantiate all detectors + aggregator + fusion. Cached."""
    face = FaceDetector()

    video = VideoDeepfakeDetector()
    video.load()

    audio = AudioDeepfakeDetector()   # stub, does nothing yet
    audio.load()

    fusion = MultimodalFusion()
    return face, video, audio, fusion


with st.spinner("Loading detection pipeline..."):
    face_detector, video_detector, audio_detector, fusion = load_pipeline()

st.success(f"✅ Pipeline ready on **{video_detector.device.upper()}**")
st.markdown("---")


# ============================================================
# HELPERS
# ============================================================
def classify(score):
    """Turn a fake-confidence score into a label + BGR color."""
    if score < config.REAL_THRESHOLD:
        return "REAL", (0, 255, 0)
    elif score < config.SUSPICIOUS_THRESHOLD:
        return "SUSPICIOUS", (0, 165, 255)
    else:
        return "FAKE", (0, 0, 255)


def process_frame(frame_bgr, aggregator, frame_count):
    """
    Run the full detection pipeline on a single frame.
    Returns the annotated frame plus a dict of metrics.
    """
    faces = face_detector.detect(frame_bgr)

    top_score = 0.0
    top_label = "NO FACE"
    inference_ms = 0.0
    fused_score = 0.0
    explanation = ""

    run_model = (frame_count % config.FRAME_SKIP == 0)

    for box in faces:
        crop = face_detector.crop_face(frame_bgr, box)
        if crop is None:
            continue

        # Run models every Nth frame; reuse cached score otherwise
        cache_key = f"last_score_{box[0]}_{box[1]}"
        if run_model:
            t0 = time.time()
            video_score = video_detector.predict(crop)
            audio_score = audio_detector.predict(None)  # stub returns None

            result = fusion.fuse({
                "video_primary": video_score,
                "audio": audio_score,
            })
            fused_score = result["fused_score"]
            explanation = fusion.explanation(result)
            inference_ms = (time.time() - t0) * 1000

            st.session_state[cache_key] = fused_score
            st.session_state[cache_key + "_expl"] = explanation
        else:
            fused_score = st.session_state.get(cache_key, 0.0)
            explanation = st.session_state.get(cache_key + "_expl", "")

        label, color = classify(fused_score)

        # Draw bounding box + label
        x, y, w, h = box
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame_bgr, f"{label} {fused_score * 100:.1f}%",
            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )

        if fused_score > top_score:
            top_score = fused_score
            top_label = label

    # Update temporal aggregator (only when a model actually ran)
    agg_result = {"smoothed_score": 0.0, "alert_fired": False}
    if run_model and len(faces) > 0:
        agg_result = aggregator.update(top_score)

    # Draw alert banner if we're currently in alert state
    if aggregator.is_currently_alerting() and len(faces) > 0:
        cv2.rectangle(frame_bgr, (0, 0), (frame_bgr.shape[1], 40), (0, 0, 255), -1)
        cv2.putText(
            frame_bgr, "!  DEEPFAKE ALERT",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )

    return frame_bgr, {
        "top_score": top_score,
        "top_label": top_label,
        "smoothed_score": agg_result.get("smoothed_score", 0.0),
        "inference_ms": inference_ms,
        "faces": len(faces),
        "explanation": explanation,
    }


def render_dashboard(placeholders, metrics, fps):
    """Push metrics into the right-side dashboard."""
    if metrics["faces"] == 0:
        placeholders["status"].info("⚪ No face detected")
        placeholders["confidence"].metric("Fake Confidence", "—")
        placeholders["smoothed"].metric("Smoothed", "—")
    else:
        emoji_map = {"REAL": "🟢", "SUSPICIOUS": "🟠", "FAKE": "🔴"}
        emoji = emoji_map.get(metrics["top_label"], "⚪")
        placeholders["status"].info(f"{emoji} **{metrics['top_label']}**")
        placeholders["confidence"].metric(
            "Fake Confidence", f"{metrics['top_score'] * 100:.1f}%"
        )
        placeholders["smoothed"].metric(
            f"Smoothed (last {config.WINDOW_SIZE})",
            f"{metrics['smoothed_score'] * 100:.1f}%",
        )

    placeholders["fps"].metric("FPS", f"{fps:.1f}")
    placeholders["latency"].metric("Model Latency", f"{metrics['inference_ms']:.0f} ms")
    placeholders["faces"].metric("Faces Detected", metrics["faces"])

    if metrics["explanation"]:
        placeholders["explanation"].caption(f"🧮  {metrics['explanation']}")


# ============================================================
# MODE SELECTOR
# ============================================================
mode = st.radio(
    "Input Source",
    ["📹 Live Webcam", "📁 Upload Video File"],
    horizontal=True,
)
st.markdown("---")

# Shared layout
col_video, col_info = st.columns([2, 1])

with col_video:
    st.subheader("Video Feed")
    video_placeholder = st.empty()

with col_info:
    st.subheader("Detection Status")
    placeholders = {
        "status":      st.empty(),
        "confidence":  st.empty(),
        "smoothed":    st.empty(),
        "fps":         st.empty(),
        "latency":     st.empty(),
        "faces":       st.empty(),
        "explanation": st.empty(),
    }
    st.markdown("---")
    st.subheader("🚨 Alert Log")
    alert_placeholder = st.empty()


# ============================================================
# SESSION STATE
# ============================================================
if "running" not in st.session_state:
    st.session_state.running = False
if "aggregator" not in st.session_state:
    st.session_state.aggregator = TemporalAggregator()


# ============================================================
# MODE HANDLERS
# ============================================================
def run_source(source, is_stream, progress_bar=None):
    """Shared frame loop for webcam and video-file sources."""
    aggregator = st.session_state.aggregator
    aggregator.reset()

    prev_time = time.time()
    frame_count = 0

    while True:
        if is_stream and not st.session_state.running:
            break

        ret, frame = source.read()
        if not ret:
            break

        annotated, metrics = process_frame(frame, aggregator, frame_count)

        curr_time = time.time()
        fps = 1 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time
        frame_count += 1

        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB")
        render_dashboard(placeholders, metrics, fps)

        # Alert log
        alerts = aggregator.recent_alerts(5)
        if not alerts:
            alert_placeholder.info("No alerts triggered yet.")
        else:
            log = "\n\n".join(
                [f"🚨 **{ts}** — score `{s * 100:.1f}%`" for ts, s in alerts]
            )
            alert_placeholder.markdown(log)

        if progress_bar is not None:
            progress_bar.progress(source.progress())


if mode == "📹 Live Webcam":
    with col_video:
        start = st.button("▶ Start Camera", type="primary", key="start_cam")
        stop = st.button("⏹ Stop Camera", key="stop_cam")

    if start:
        st.session_state.running = True
    if stop:
        st.session_state.running = False

    if st.session_state.running:
        try:
            with WebcamSource() as cam:
                run_source(cam, is_stream=True)
        except RuntimeError as e:
            st.error(f"❌ {e}")
    else:
        video_placeholder.info("👆 Click 'Start Camera' to begin.")
        placeholders["status"].info("⚪ Idle")


elif mode == "📁 Upload Video File":
    with col_video:
        uploaded = st.file_uploader(
            "Upload a video (mp4, mov, avi, webm)",
            type=["mp4", "mov", "avi", "webm"],
        )
        playback = st.radio(
            "Playback mode",
            ["⚡ Real-time (skip frames)", "🔬 Full analysis (every frame)"],
            key="playback",
        )

    if uploaded is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        tfile.close()

        st.success(f"✅ Loaded: {uploaded.name}")

        with col_video:
            analyze = st.button("▶ Analyze Video", type="primary", key="analyze")

        if analyze:
            playback_mode = "real_time" if "Real-time" in playback else "full"
            try:
                with VideoFileSource(tfile.name, playback_mode=playback_mode) as vsrc:
                    info = vsrc.info()
                    st.info(
                        f"Video: {info['video_fps']:.1f} FPS, "
                        f"{info['total_frames']} frames · "
                        f"reading every {info['read_skip']} frame(s)"
                    )
                    progress = st.progress(0)
                    run_source(vsrc, is_stream=False, progress_bar=progress)
                    progress.progress(1.0)
                st.success("✅ Analysis complete.")
            except RuntimeError as e:
                st.error(f"❌ {e}")
    else:
        video_placeholder.info("👆 Upload a video file to analyze.")
        placeholders["status"].info("⚪ Idle")