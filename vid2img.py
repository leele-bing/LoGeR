from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2


def list_video_files(video_root: str | Path) -> List[Path]:
    root = Path(video_root).expanduser().resolve()
    if root.is_file():
        return [root]
    if root.is_dir():
        return sorted(root.glob("*.mp4"))
    return []


def sample_video_frames(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    target_fps: float = 3.0,
    max_minutes: float = 8.0,
) -> Dict[str, Any]:
    video_path = Path(video_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_seconds = max_minutes * 60.0
    if total_frames > 0:
        max_seconds = min(max_seconds, total_frames / fps)

    sample_period = 1.0 / target_fps
    next_sample_ts = 0.0
    frame_idx = 0
    saved_idx = 0
    frame_paths: List[str] = []

    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_ts = frame_idx / fps
        if current_ts > max_seconds + 1e-6:
            break

        if current_ts + 1e-6 >= next_sample_ts:
            frame_path = output_dir / f"{saved_idx:06d}.png"
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"Failed to save frame to {frame_path}")
            frame_paths.append(str(frame_path))
            saved_idx += 1
            next_sample_ts += sample_period

        frame_idx += 1

    cap.release()

    return {
        "video_path": str(video_path),
        "output_dir": str(output_dir),
        "frame_paths": frame_paths,
        "num_frames": saved_idx,
        "cached": False,
        "sampling_seconds": round(time.time() - start_time, 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample videos into frame folders.")
    parser.add_argument(
        "--video_root",
        type=str,
        default="/home/pcl/Dataset/UrbanNav_videos",
        help="Directory containing mp4 videos, or a single video file path.",
    )
    parser.add_argument("--output_root", type=str, default="data", help="Directory to store sampled frame folders.")
    parser.add_argument("--max_videos", type=int, default=5, help="Number of videos to process from the sorted source list.")
    parser.add_argument("--target_fps", type=float, default=3.0, help="Sampling FPS.")
    parser.add_argument("--max_minutes", type=float, default=5.0, help="Maximum duration to sample from each video.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    videos = list_video_files(args.video_root)[: args.max_videos]
    if not videos:
        raise FileNotFoundError(f"No valid video input found in {args.video_root}")

    print(f"Sampling {len(videos)} video(s) from {args.video_root}")
    for index, video_path in enumerate(videos, start=1):
        output_dir = Path(args.output_root) / video_path.stem
        result = sample_video_frames(
            video_path,
            output_dir,
            target_fps=args.target_fps,
            max_minutes=args.max_minutes,
        )
        print(
            f"[{index}/{len(videos)}] {video_path.name}: sampled, "
            f"{result['num_frames']} frames -> {result['output_dir']}"
        )


if __name__ == "__main__":
    main()
