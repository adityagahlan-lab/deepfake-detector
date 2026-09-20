"""
Extract face crops from FaceForensics++ videos.

Adapted for the Kaggle `xdxd003/ff-c23` layout (flatter than official FF++):

    FaceForensics++_C23/
    ├── original/*.mp4              ← real videos
    ├── Deepfakes/*.mp4             ← fake method 1
    ├── Face2Face/*.mp4             ← fake method 2
    ├── FaceShifter/*.mp4           ← fake method 3
    ├── FaceSwap/*.mp4              ← fake method 4
    └── NeuralTextures/*.mp4        ← fake method 5

Output layout (per-video subfolders — LSTM-compatible):

    faces/
    ├── real/
    │   ├── 000/                    (000.jpg, 001.jpg, ...)
    │   └── ...
    └── fake/
        ├── 000_Deepfakes/          (000.jpg, 001.jpg, ...)
        ├── 000_Face2Face/          ...
        └── ...

Safe to interrupt and re-run — skips folders already fully processed.
"""
import argparse
import random
from pathlib import Path

import cv2
from tqdm import tqdm


FRAME_STRIDE = 10        # every Nth frame (was 5; 10 halves CPU time)
FACE_INPUT   = 224
MIN_FACES    = 8         # skip videos with fewer than this many face crops


def get_face_detector():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )


def extract_from_video(video_path: Path, out_dir: Path, detector) -> int:
    if out_dir.exists() and any(out_dir.glob("*.jpg")):
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

    if saved_idx < MIN_FACES:
        for f in out_dir.glob("*.jpg"):
            f.unlink()
        try:
            out_dir.rmdir()
        except OSError:
            pass
        return 0

    return saved_idx


def process_folder(video_dir: Path, output_root: Path, label: str,
                   method_tag: str, detector, limit: int = None,
                   seed: int = 42):
    videos = sorted(video_dir.glob("*.mp4"))
    if not videos:
        print(f"[warn] no .mp4 files in {video_dir}")
        return 0, 0

    if limit and len(videos) > limit:
        # Deterministic random sample so re-runs pick the same videos
        rng = random.Random(seed)
        videos = rng.sample(videos, limit)
        videos.sort()

    print(f"\n[{label} / {method_tag or 'original'}] {len(videos)} videos")

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
    parser = argparse.ArgumentParser(description="Extract face crops (Kaggle FF++ layout)")
    parser.add_argument("--data-root",   required=True,
                        help="Path to FaceForensics++_C23 folder")
    parser.add_argument("--output-root", required=True,
                        help="Where to write face crops (e.g. training/faces/)")
    parser.add_argument("--real-limit",  type=int, default=1000,
                        help="Max number of real videos to process")
    parser.add_argument("--per-method-limit", type=int, default=250,
                        help="Max fake videos per manipulation method")
    parser.add_argument("--methods", nargs="+",
                        default=["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"],
                        help="Which fake-generation methods to include")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    detector = get_face_detector()

    # ------- REAL -------
    real_dir = data_root / "original"
    if not real_dir.exists():
        print(f"[error] {real_dir} not found. Check --data-root path.")
        return
    n_ok_real, n_faces_real = process_folder(
        real_dir, output_root, "real", "", detector, limit=args.real_limit,
    )

    # ------- FAKE (each method) -------
    n_ok_fake = 0
    n_faces_fake = 0
    for method in args.methods:
        method_dir = data_root / method
        if not method_dir.exists():
            print(f"[warn] method folder missing: {method_dir}")
            continue
        n_ok, n_faces = process_folder(
            method_dir, output_root, "fake", method, detector,
            limit=args.per_method_limit,
        )
        n_ok_fake += n_ok
        n_faces_fake += n_faces

    print("\n" + "=" * 60)
    print(f"REAL: {n_ok_real} videos processed, {n_faces_real:,} face crops")
    print(f"FAKE: {n_ok_fake} videos processed, {n_faces_fake:,} face crops")
    print(f"Output: {output_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()