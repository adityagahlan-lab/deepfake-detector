import streamlit as st
import cv2
import time
import tempfile
import torch
from collections import deque
from datetime import datetime
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


# Page config
st.set_page_config(
    page_title="Deepfake Detection Platform",
    page_icon="🎭",
    layout="wide"
)

# Header
st.title("🎭 Real-Time Deepfake Detection Platform")
st.caption("SIH 2026 | USICT028 | AI Safety & Media Verification")
st.markdown("---")
# ============================================================
# SIDEBAR - project info
# ============================================================
with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/2103/2103633.png",
        width=80
    )
    st.markdown("### About")
    st.markdown(
        "A real-time deepfake detection platform for live video streams. "
        "Analyzes each frame through a Vision Transformer, "
        "aggregates predictions temporally, and triggers alerts on sustained "
        "high-confidence detections."
    )

    st.markdown("### Tech Stack")
    st.markdown(
        "- **Python 3.13** + PyTorch\n"
        "- **HuggingFace Transformers** (ViT deepfake model)\n"
        "- **OpenCV** (frame capture + face detection)\n"
        "- **Streamlit** (dashboard)\n"
    )

    st.markdown("### Pipeline")
    st.markdown(
        "1. Video capture\n"
        "2. Face detection (Haar Cascade)\n"
        "3. Frame-level classification (ViT)\n"
        "4. Temporal smoothing (rolling window)\n"
        "5. Alert on N-frame streak"
    )

    st.markdown("### Team")
    st.markdown("**[Your team name]**")
    st.markdown("SIH 2026 | USICT028")

    st.markdown("---")
    st.caption("⚠️ Prototype running on CPU. Production deployment targets GPU inference (~30ms/frame).")


# ============================================================
# CONFIG
# ============================================================
WINDOW_SIZE = 5
ALERT_THRESHOLD = 0.6
CONSECUTIVE_ALERTS_NEEDED = 3
FRAME_SKIP = 1  # TODO: set to 1 on GPU


# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    try:
        model_name = "prithivMLmods/Deep-Fake-Detector-Model"
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        return processor, model, device
    except Exception as e:
        st.error(f"❌ Model load failed: {e}")
        raise


with st.spinner("Loading deepfake detection model..."):
    processor, model, device = load_model()

st.success(f"✅ Model loaded on **{device.upper()}**")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ============================================================
# DEEPFAKE DETECTOR
# ============================================================
def deepfake_detector(face_bgr):
    face_bgr = cv2.resize(face_bgr, (224, 224))
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(face_rgb)

    inputs = processor(images=pil_image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    labels = model.config.id2label
    fake_idx = next(i for i, name in labels.items() if "fake" in name.lower())
    return probs[fake_idx].item()


def classify(score):
    if score < 0.4:
        return "REAL", (0, 255, 0)
    elif score < 0.7:
        return "SUSPICIOUS", (0, 165, 255)
    else:
        return "FAKE", (0, 0, 255)


# ============================================================
# MODE SELECTOR (top of page)
# ============================================================
mode = st.radio(
    "Input Source",
    ["📹 Live Webcam", "📁 Upload Video File"],
    horizontal=True,
)

st.markdown("---")


# ============================================================
# CORE PROCESSING FUNCTION (shared by both modes)
# ============================================================
def process_video_source(cap, video_placeholder, status_placeholder,
                        confidence_placeholder, smoothed_placeholder,
                        fps_placeholder, latency_placeholder, faces_placeholder,
                        alert_log_placeholder, stop_check=lambda: False,
                        read_skip=1, progress_bar=None, total_frames=0):
    """
    Reads frames from `cap` and runs the full detection pipeline.
    stop_check: callable that returns True when we should stop the loop.
    """
    prev_time = time.time()
    frame_count = 0
    score_history = deque(maxlen=WINDOW_SIZE)
    alert_streak = 0
    alerts = []
    last_scores = {}  # cache per-face last scores

    processed_count = 0
    while not stop_check():
        # Skip N-1 frames without decoding (fast), then read the Nth
        for _ in range(read_skip - 1):
            cap.grab()
            processed_count += 1

        ret, frame = cap.read()
        processed_count += 1
        if not ret:
            break  # end of video or webcam failure

        if progress_bar is not None and total_frames > 0:
            progress_bar.progress(min(processed_count / total_frames, 1.0))

        frame = cv2.flip(frame, 1) if not st.session_state.get("is_file", False) else frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        top_score = 0.0
        top_label = "NO FACE"
        inference_latency = 0.0
        run_model_this_frame = (frame_count % FRAME_SKIP == 0)

        for (x, y, w, h) in faces:
            face_crop = frame[y:y + h, x:x + w]
            if face_crop.size == 0:
                continue

            if run_model_this_frame:
                inference_start = time.time()
                score = deepfake_detector(face_crop)
                inference_latency = (time.time() - inference_start) * 1000
                last_scores[(x, y)] = score
            else:
                score = last_scores.get((x, y), 0.0)

            label, color = classify(score)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {score * 100:.1f}%", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if score > top_score:
                top_score = score
                top_label = label

        # Temporal aggregation
        if run_model_this_frame and len(faces) > 0:
            score_history.append(top_score)

        smoothed_score = sum(score_history) / len(score_history) if score_history else 0.0

        # Alert logic
        if run_model_this_frame and len(faces) > 0:
            if smoothed_score >= ALERT_THRESHOLD:
                alert_streak += 1
            else:
                alert_streak = 0

            if alert_streak >= CONSECUTIVE_ALERTS_NEEDED:
                timestamp = datetime.now().strftime("%H:%M:%S")
                if not alerts or alerts[-1][0] != timestamp:
                    alerts.append((timestamp, smoothed_score))
                alert_streak = 0

        # Alert banner on frame
        if smoothed_score >= ALERT_THRESHOLD and len(faces) > 0:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 255), -1)
            cv2.putText(frame, "! DEEPFAKE ALERT", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        frame_count += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB")

        if len(faces) == 0:
            status_placeholder.info("⚪ No face detected")
            confidence_placeholder.metric("Fake Confidence", "—")
            smoothed_placeholder.metric("Smoothed Score", "—")
        else:
            emoji_map = {"REAL": "🟢", "SUSPICIOUS": "🟠", "FAKE": "🔴"}
            emoji = emoji_map.get(top_label, "⚪")
            status_placeholder.info(f"{emoji} Status: **{top_label}**")
            confidence_placeholder.metric("Fake Confidence", f"{top_score * 100:.1f}%")
            smoothed_placeholder.metric(f"Smoothed (last {WINDOW_SIZE})",
                                        f"{smoothed_score * 100:.1f}%")

        fps_placeholder.metric("FPS", f"{fps:.1f}")
        latency_placeholder.metric("Model Latency", f"{inference_latency:.0f} ms")
        faces_placeholder.metric("Faces Detected", len(faces))

        if not alerts:
            alert_log_placeholder.info("No alerts triggered yet.")
        else:
            recent = alerts[-5:][::-1]
            log_text = "\n\n".join([f"🚨 **{ts}** — score `{s * 100:.1f}%`" for ts, s in recent])
            alert_log_placeholder.markdown(log_text)

    cap.release()


# ============================================================
# LAYOUT (shared between both modes)
# ============================================================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Video Feed")
    video_placeholder = st.empty()

with col2:
    st.subheader("Detection Status")
    status_placeholder = st.empty()
    confidence_placeholder = st.empty()
    smoothed_placeholder = st.empty()
    fps_placeholder = st.empty()
    latency_placeholder = st.empty()
    faces_placeholder = st.empty()
    st.markdown("---")
    st.subheader("🚨 Alert Log")
    alert_log_placeholder = st.empty()
    st.markdown("---")
    st.markdown(f"**Config:** window={WINDOW_SIZE}, threshold={ALERT_THRESHOLD}, streak={CONSECUTIVE_ALERTS_NEEDED}")


# Session state
if "running" not in st.session_state:
    st.session_state.running = False


# ============================================================
# MODE: LIVE WEBCAM
# ============================================================
if mode == "📹 Live Webcam":
    st.session_state.is_file = False

    with col1:
        start_button = st.button("▶ Start Camera", type="primary", key="start_webcam")
        stop_button = st.button("⏹ Stop Camera", key="stop_webcam")

    if start_button:
        st.session_state.running = True
    if stop_button:
        st.session_state.running = False

    if st.session_state.running:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        if not cap.isOpened():
            st.error("❌ Could not open webcam.")
        else:
            process_video_source(
                cap, video_placeholder, status_placeholder,
                confidence_placeholder, smoothed_placeholder,
                fps_placeholder, latency_placeholder, faces_placeholder,
                alert_log_placeholder,
                stop_check=lambda: not st.session_state.running,
            )
    else:
        video_placeholder.info("👆 Click 'Start Camera' to begin.")
        status_placeholder.info("⚪ Idle")


# ============================================================
# MODE: UPLOAD VIDEO FILE
# ============================================================
elif mode == "📁 Upload Video File":
    st.session_state.is_file = True

    with col1:
        uploaded_file = st.file_uploader(
            "Upload a video (mp4, mov, avi, webm)",
            type=["mp4", "mov", "avi", "webm"]
        )
        playback_mode = st.radio(
            "Playback mode",
            ["⚡ Real-time (skip frames to match video speed)",
             "🔬 Full analysis (slow, analyzes every frame)"],
            key="playback_mode"
        )

    if uploaded_file is not None:
        # Save uploaded file to a temp path (OpenCV needs a real path)
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()

        st.success(f"✅ Loaded: {uploaded_file.name}")

        analyze_button = st.button("▶ Analyze Video", type="primary", key="analyze_file")

        if analyze_button:
            cap = cv2.VideoCapture(tfile.name)
            if not cap.isOpened():
                st.error("❌ Could not open video file.")
            else:
                video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                if "Real-time" in playback_mode:
                    pipeline_fps = 2  # our CPU pipeline speed
                    read_skip = max(1, int(video_fps / pipeline_fps))
                else:
                    read_skip = 1  # analyze every frame

                st.info(
                    f"Video: {video_fps:.1f} FPS, {total_frames} frames | "
                    f"Reading every {read_skip} frame(s)"
                )

                progress_bar = st.progress(0)

                process_video_source(
                    cap, video_placeholder, status_placeholder,
                    confidence_placeholder, smoothed_placeholder,
                    fps_placeholder, latency_placeholder, faces_placeholder,
                    alert_log_placeholder,
                    stop_check=lambda: False,
                    read_skip=read_skip,
                    progress_bar=progress_bar,
                    total_frames=total_frames,
                )
                progress_bar.progress(1.0)
                st.success("✅ Analysis complete.")
    else:
        video_placeholder.info("👆 Upload a video file to analyze.")
        status_placeholder.info("⚪ Idle")