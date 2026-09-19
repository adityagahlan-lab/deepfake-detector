"""
Extract face crops from FaceForensics++ videos.

Expected FF++ folder layout (matches the official dataset structure):

    data/
    ├── original_sequences/
    │   └── youtube/c23/videos/*.mp4         # REAL videos
    └── manipulated_sequences/
        ├── Deepfakes/c23/videos/*.mp4       # FAKE, method 1
        ├── Face2Face/c23/videos/*.mp4       # FAKE, method 2
        ├── FaceSwap/c23/videos/*.mp4        # FAKE, method 3
        └── NeuralTextures/c23/videos/*.mp4  # FAKE, method 4

Output layout:

    faces/
    ├── real/
    │   ├── 000/000.jpg  000/001.jpg  ...    # one folder per video
    │   └── ...
    └── fake/
        ├── 000_Deepfakes/000.jpg  ...
        ├── 000_Face2Face/000.jpg  ...
        └── ...

Runtime notes:
  * Uses OpenCV's Haar cascade for face detection (fast, CPU-friendly)
  * Sampling every Nth frame (default 5) to control dataset size
  * Skips videos where fewer than MIN_FACES faces are detected
  * Safe to interrupt and re-run — skips videos already processed
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
from tqdm import tqdm


FRAME_STRIDE   = 5        # process every 5th frame
FACE_INPUT     = 224      # output crop size (matches ViT input)
MIN_FACES      = 10       # videos with < 10 detected faces are skipped


def get_face_detector():
    """OpenCV Haar cascade — fast enough for dataset prep."""
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )


def extract_from_video(video_path: Path, out_dir: Path, detector) -> int:
    """
    Extract face crops from a single video. Returns count of saved faces.
    Skips silently if out_dir already exists AND has crops.
    """
    if out_dir.exists() and any(out_dir.glob("*.jpg")):
        # Already processed — count and return
        return len(list(out_dir.glob("*.jpg")))

    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    frame_idx = 0
    saved_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_STRIDE == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            if len(faces) > 0:
                # Take the largest face in the frame
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                crop = frame[y:y + h, x:x + w]
                if crop.size > 0:
                    crop_resized = cv2.resize(crop, (FACE_INPUT, FACE_INPUT))
                    cv2.imwrite(
                        str(out_dir / f"{saved_idx:04d}.jpg"),
                        crop_resized,
                        [cv2.IMWRITE_JPEG_QUALITY, 90],
                    )
                    saved_idx += 1

        frame_idx += 1

    cap.release()

    # If too few faces found, remove the folder so caller knows to skip
    if saved_idx < MIN_FACES:
        for f in out_dir.glob("*.jpg"):
            f.unlink()
        try:
            out_dir.rmdir()
        except OSError:
            pass
        return 0

    return saved_idx


def process_directory(video_root: Path,
                      output_root: Path,
                      label: str,
                      method_tag: str,
                      detector,
                      max_videos: int = None):
    """
    Process every video in video_root and save face crops under output_root/<label>/.

    label      : "real" or "fake" — determines top-level output subfolder
    method_tag : appended to output folder names ("" for real, "Deepfakes" etc. for fake)
    max_videos : optional cap for quick testing (None = all)
    """
    videos = sorted(video_root.glob("*.mp4"))
    if max_videos:
        videos = videos[:max_videos]

    if not videos:
        print(f"[warn] No .mp4 files in {video_root}")
        return 0, 0

    print(f"\n[{label} / {method_tag or 'youtube'}] {len(videos)} videos in {video_root}")

    n_ok = 0
    n_faces_total = 0
    for video in tqdm(videos, unit="video"):
        stem = video.stem
        subfolder = f"{stem}_{method_tag}" if method_tag else stem
        out_dir = output_root / label / subfolder
        n_faces = extract_from_video(video, out_dir, detector)
        if n_faces > 0:
            n_ok += 1
            n_faces_total += n_faces

    return n_ok, n_faces_total


def main():
    parser = argparse.ArgumentParser(description="Extract face crops from FaceForensics++")
    parser.add_argument("--data-root",   required=True,
                        help="Path to FF++ data root (containing original_sequences/ and manipulated_sequences/)")
    parser.add_argument("--output-root", required=True,
                        help="Where to write face crops (e.g. training/faces/)")
    parser.add_argument("--max-videos",  type=int, default=None,
                        help="Cap the number of videos per category (for quick testing)")
    parser.add_argument("--methods", nargs="+",
                        default=["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"],
                        help="Which fake-generation methods to include")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    detector = get_face_detector()

    # --- REAL videos ---
    real_dir = data_root / "original_sequences" / "youtube" / "c23" / "videos"
    n_ok_real, n_faces_real = process_directory(
        real_dir, output_root, label="real",
        method_tag="", detector=detector, max_videos=args.max_videos,
    )

    # --- FAKE videos (one folder per method) ---
    n_ok_fake_total = 0
    n_faces_fake_total = 0
    for method in args.methods:
        fake_dir = data_root / "manipulated_sequences" / method / "c23" / "videos"
        if not fake_dir.exists():
            print(f"[warn] Method folder missing: {fake_dir}")
            continue
        n_ok, n_faces = process_directory(
            fake_dir, output_root, label="fake",
            method_tag=method, detector=detector, max_videos=args.max_videos,
        )
        n_ok_fake_total += n_ok
        n_faces_fake_total += n_faces

    print("\n" + "=" * 60)
    print(f"REAL: {n_ok_real} videos processed, {n_faces_real:,} face crops")
    print(f"FAKE: {n_ok_fake_total} videos processed, {n_faces_fake_total:,} face crops")
    print(f"Output written to: {output_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()