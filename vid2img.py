from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2


def list_video_files(video_root: str | Path) -> List[Path]:
    root = Path(video_root).expanduser().resolve()
    if root.is_file():
        return [root]
    if root.is_dir():
        return sorted(root.glob("*.mp4"))
    return []


def list_existing_frame_files(output_dir: str | Path) -> List[Path]:
    root = Path(output_dir).expanduser().resolve()
    if not root.is_dir():
        return []
    return sorted(
        (path for path in root.rglob("*.jpg") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def _collect_output_dirs(root_dir: Path, target_frames: int | None) -> List[str]:
    if target_frames is None:
        return [str(root_dir)]
    segment_dirs = sorted(path for path in root_dir.iterdir() if path.is_dir())
    return [str(path) for path in segment_dirs]


def _write_frame_copy(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        target_path.hardlink_to(source_path)
        return
    except Exception:
        pass
    shutil.copy2(source_path, target_path)


def _chunk_frames(frame_paths: List[Path], target_frames: int | None, min_last_group_frames: int = 0) -> List[List[Path]]:
    if target_frames is None:
        return [frame_paths]
    if target_frames <= 0:
        raise ValueError(f"target_frames must be positive, got {target_frames}")

    groups = [frame_paths[start : start + target_frames] for start in range(0, len(frame_paths), target_frames)]
    if min_last_group_frames > 0 and len(groups) >= 2 and len(groups[-1]) < min_last_group_frames:
        groups[-2].extend(groups[-1])
        groups.pop()
    return groups


def _merge_small_tail_group(root_dir: Path, target_frames: int | None, min_last_group_frames: int) -> None:
    if target_frames is None or min_last_group_frames <= 0:
        return
    segment_dirs = sorted(path for path in root_dir.iterdir() if path.is_dir())
    if len(segment_dirs) < 2:
        return

    last_dir = segment_dirs[-1]
    prev_dir = segment_dirs[-2]
    last_files = sorted(
        (path for path in last_dir.iterdir() if path.is_file() and path.suffix.lower() == ".jpg"),
        key=lambda path: path.name,
    )
    if len(last_files) >= min_last_group_frames:
        return

    prev_count = sum(1 for path in prev_dir.iterdir() if path.is_file() and path.suffix.lower() == ".jpg")
    for offset, source_path in enumerate(last_files):
        target_path = prev_dir / f"{prev_count + offset:06d}.jpg"
        source_path.rename(target_path)
    shutil.rmtree(last_dir)


def rebuild_from_existing_frames(
    output_dir: str | Path,
    *,
    video_path: str | Path,
    target_fps: float,
    max_minutes: Optional[float],
    target_frames: int | None,
    min_last_group_frames: int = 0,
    jpeg_quality: int,
) -> Dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    existing_frames = list_existing_frame_files(output_dir)
    if not existing_frames:
        raise FileNotFoundError(f"No existing frames found under {output_dir}")

    _ = video_path
    _ = target_fps
    _ = max_minutes
    selected_frames = existing_frames
    frame_groups = _chunk_frames(selected_frames, target_frames, min_last_group_frames=min_last_group_frames)

    temp_output_dir = output_dir.parent / f".{output_dir.name}.overwrite_tmp"
    if temp_output_dir.exists():
        shutil.rmtree(temp_output_dir)
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    saved_idx = 0
    output_dirs: List[str] = []
    for segment_idx, group in enumerate(frame_groups):
        if target_frames is None:
            segment_dir = temp_output_dir
        else:
            segment_dir = temp_output_dir / f"{output_dir.name}_{segment_idx:03d}"
            output_dirs.append(str(segment_dir))
        for segment_frame_idx, source_path in enumerate(group):
            target_path = segment_dir / f"{segment_frame_idx:06d}.jpg"
            _write_frame_copy(source_path, target_path)
            saved_idx += 1

    if output_dir.exists():
        shutil.rmtree(output_dir)
    temp_output_dir.rename(output_dir)

    return {
        "video_path": str(Path(video_path).expanduser().resolve()),
        "output_dir": str(output_dir),
        "output_dirs": output_dirs if target_frames is not None else [str(output_dir)],
        "frame_paths": [str(path) for path in list_existing_frame_files(output_dir)],
        "num_frames": len(selected_frames),
        "cached": True,
        "reused_existing_frames": True,
        "sampling_seconds": round(time.time() - start_time, 3),
    }


def sample_video_frames(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    target_fps: float = 3.0,
    max_minutes: Optional[float] = None,
    target_frames: int | None = None,
    min_last_group_frames: int = 0,
    jpeg_quality: int = 95,
) -> Dict[str, Any]:
    video_path = Path(video_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if target_frames is not None and target_frames <= 0:
        raise ValueError(f"target_frames must be positive, got {target_frames}")
    if max_minutes is not None and max_minutes <= 0:
        raise ValueError(f"max_minutes must be positive when provided, got {max_minutes}")
    if not (0 <= int(jpeg_quality) <= 100):
        raise ValueError(f"jpeg_quality must be between 0 and 100, got {jpeg_quality}")

    base_output_dir = output_dir

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_minutes is None:
        max_seconds = float("inf") if total_frames <= 0 else total_frames / fps
    else:
        max_seconds = max_minutes * 60.0
        if total_frames > 0:
            max_seconds = min(max_seconds, total_frames / fps)

    sample_period = 1.0 / target_fps
    next_sample_ts = 0.0
    frame_idx = 0
    saved_idx = 0
    frame_paths: List[str] = []
    output_dirs: List[str] = []

    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_ts = frame_idx / fps
        if current_ts > max_seconds + 1e-6:
            break

        if current_ts + 1e-6 >= next_sample_ts:
            if target_frames is None:
                segment_dir = base_output_dir
                segment_dir.mkdir(parents=True, exist_ok=True)
                segment_frame_idx = saved_idx
            else:
                segment_idx = saved_idx // target_frames
                segment_dir = base_output_dir / f"{base_output_dir.name}_{segment_idx:03d}"
                segment_dir.mkdir(parents=True, exist_ok=True)
                segment_frame_idx = saved_idx % target_frames
                segment_dir_str = str(segment_dir)
                if not output_dirs or output_dirs[-1] != segment_dir_str:
                    output_dirs.append(segment_dir_str)

            frame_path = segment_dir / f"{segment_frame_idx:06d}.jpg"
            if not cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]):
                raise RuntimeError(f"Failed to save frame to {frame_path}")
            frame_paths.append(str(frame_path))
            saved_idx += 1
            next_sample_ts += sample_period

        frame_idx += 1

    cap.release()

    _merge_small_tail_group(base_output_dir, target_frames, min_last_group_frames)
    frame_paths = [str(path) for path in list_existing_frame_files(base_output_dir)]
    output_dirs = _collect_output_dirs(base_output_dir, target_frames)

    return {
        "video_path": str(video_path),
        "output_dir": str(base_output_dir),
        "output_dirs": output_dirs,
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
        default="/data/xby/YTB/bangkok/vid",
        help="Directory containing mp4 videos, or a single video file path.",
    )
    parser.add_argument("--output_root", type=str, default="/data/xby/YTB/bangkok/img", help="Directory to store sampled frame folders.")
    parser.add_argument("--max_videos", type=int, default=-1, help="Number of videos to process from the sorted source list.")
    parser.add_argument("--target_fps", type=float, default=3.0, help="Sampling FPS.")
    parser.add_argument("--max_minutes", type=float, default=None, help="Maximum duration to sample from each video. Default: no duration limit.")
    parser.add_argument(
        "--target_frames",
        type=int,
        default=None,
        help="Maximum number of frames per output subfolder. When set, frames are stored under <video>/<video>_000, <video>/<video>_001, ...",
    )
    parser.add_argument(
        "--min_last_group_frames",
        type=int,
        default=360,
        help="Merge the last group into the previous one if it has fewer than this many frames. Use 0 to disable.",
    )
    parser.add_argument("--jpeg_quality", type=int, default=95, help="JPEG quality for saved frames, from 0 to 100.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If output frames already exist, rebuild the folder/chunk layout from the existing sampled images instead of decoding the video again.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_videos = list_video_files(args.video_root)
    videos = all_videos if args.max_videos < 0 else all_videos[: args.max_videos]
    if not videos:
        raise FileNotFoundError(f"No valid video input found in {args.video_root}")

    print(f"Sampling {len(videos)} video(s) from {args.video_root}")
    for index, video_path in enumerate(videos, start=1):
        output_dir = Path(args.output_root) / video_path.stem
        existing_frames = list_existing_frame_files(output_dir)
        if existing_frames and not args.overwrite:
            result = {
                "video_path": str(video_path),
                "output_dir": str(output_dir.resolve()),
                "output_dirs": _collect_output_dirs(output_dir.resolve(), args.target_frames),
                "frame_paths": [str(path) for path in existing_frames],
                "num_frames": len(existing_frames),
                "cached": True,
                "reused_existing_frames": False,
                "sampling_seconds": 0.0,
            }
            status = "skipped_existing"
        elif existing_frames and args.overwrite:
            result = rebuild_from_existing_frames(
                output_dir,
                video_path=video_path,
                target_fps=args.target_fps,
                max_minutes=args.max_minutes,
                target_frames=args.target_frames,
                min_last_group_frames=args.min_last_group_frames,
                jpeg_quality=args.jpeg_quality,
            )
            status = "rebuilt_from_existing"
        else:
            result = sample_video_frames(
                video_path,
                output_dir,
                target_fps=args.target_fps,
                max_minutes=args.max_minutes,
                target_frames=args.target_frames,
                min_last_group_frames=args.min_last_group_frames,
                jpeg_quality=args.jpeg_quality,
            )
            status = "sampled"
        target_desc = ", ".join(result["output_dirs"])
        print(
            f"[{index}/{len(videos)}] {video_path.name}: {status}, "
            f"{result['num_frames']} frames -> {target_desc}"
        )


if __name__ == "__main__":
    main()
