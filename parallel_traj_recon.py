from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
REQUIRED_RESULT_FILES = (
    "meta.yaml",
    "camera_poses.npz",
    "depth_maps.npz",
    "points.pt",
    "conf.npz",
    "trajectory_xz.png",
)


@dataclass(frozen=True)
class ReconTask:
    frame_dir: Path
    output_parent: Path
    output_dir: Path
    relative_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel launcher for traj_recon.py across multiple GPUs and CPU threads."
    )
    parser.add_argument("--sample_root", type=str, required=True, help="Root img directory, e.g. /data/xby/YTB/shanghai0/img")
    parser.add_argument("--output_root", type=str, required=True, help="Root traj directory, e.g. /data/xby/YTB/shanghai0/traj")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--workers_per_gpu", type=int, default=1, help="Number of concurrent recon workers per GPU.")
    parser.add_argument("--cpu_threads", type=int, default=4, help="OMP/MKL/OpenBLAS threads per worker process.")
    parser.add_argument("--max_tasks", type=int, default=None, help="Optional limit on number of leaf frame folders.")
    parser.add_argument("--log_dir", type=str, default=None, help="Optional directory to save one log file per task.")
    parser.add_argument("--dry_run", action="store_true", help="Print planned tasks without launching reconstruction.")

    parser.add_argument("--model_name", type=str, default="ckpts/Pi3X", help="Local HF Pi3X dir or local LoGeR checkpoint.")
    parser.add_argument("--config", type=str, default="ckpts/LoGeR_star/original_config.yaml", help="LoGeR config path.")
    parser.add_argument("--window_size", type=int, default=32, help="Window size for chunked inference.")
    parser.add_argument("--overlap_size", type=int, default=3, help="Overlap size between windows.")
    parser.add_argument("--reset_every", type=int, default=None, help="Reset interval for merge semantics.")
    parser.add_argument("--stride", type=int, default=3, help="Save every Nth frame result.")
    parser.add_argument("--conf_threshold", type=float, default=30.0, help="Confidence threshold forwarded to traj_recon.py.")
    parser.add_argument(
        "--resolution",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=(672, 378),
        help="Target resolution override.",
    )
    parser.add_argument("--force_annotation", action="store_true", help="Overwrite existing results.")
    parser.add_argument("--use_multimodal", action="store_true", help="Enable Pi3X multimodal branch.")
    parser.add_argument("--sim3", action="store_true", help="Use Sim3 alignment.")
    parser.add_argument("--se3", action="store_true", help="Use SE3 alignment.")
    parser.add_argument("--no_ttt", action="store_true", help="Disable TTT.")
    parser.add_argument("--no_swa", action="store_true", help="Disable SWA.")
    parser.add_argument(
        "--sim3_scale_mode",
        type=str,
        default="median",
        choices=["median", "trimmed_mean", "median_all", "trimmed_mean_all", "sim3_avg1"],
        help="Scale estimation mode for Sim3 merge.",
    )
    return parser.parse_args()


def has_images(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    return any(path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES for path in directory.iterdir())


def list_leaf_image_dirs(sample_root: Path) -> List[Path]:
    sample_root = sample_root.expanduser().resolve()
    if not sample_root.is_dir():
        return []
    if has_images(sample_root):
        return [sample_root]

    matches = [path for path in sample_root.rglob("*") if path.is_dir() and has_images(path)]
    return sorted(matches)


def result_is_complete(output_dir: Path) -> bool:
    return all((output_dir / name).exists() for name in REQUIRED_RESULT_FILES)


def build_tasks(sample_root: Path, output_root: Path, *, force_annotation: bool) -> List[ReconTask]:
    tasks: List[ReconTask] = []
    for frame_dir in list_leaf_image_dirs(sample_root):
        relative_path = frame_dir.relative_to(sample_root)
        output_dir = output_root / relative_path
        if output_dir.exists() and result_is_complete(output_dir) and not force_annotation:
            continue

        output_parent = output_dir.parent
        tasks.append(
            ReconTask(
                frame_dir=frame_dir,
                output_parent=output_parent,
                output_dir=output_dir,
                relative_name=relative_path.as_posix(),
            )
        )
    return tasks


def build_command(args: argparse.Namespace, task: ReconTask) -> List[str]:
    repo_root = Path(__file__).resolve().parent
    cmd = [
        sys.executable,
        str(repo_root / "traj_recon.py"),
        "--sample_root",
        str(task.frame_dir),
        "--output_root",
        str(task.output_parent),
        "--model_name",
        args.model_name,
        "--config",
        args.config,
        "--window_size",
        str(args.window_size),
        "--overlap_size",
        str(args.overlap_size),
        "--stride",
        str(args.stride),
        "--conf_threshold",
        str(args.conf_threshold),
        "--resolution",
        str(args.resolution[0]),
        str(args.resolution[1]),
        "--sim3_scale_mode",
        args.sim3_scale_mode,
    ]
    if args.reset_every is not None:
        cmd.extend(["--reset_every", str(args.reset_every)])
    if args.force_annotation:
        cmd.append("--force_annotation")
    if args.use_multimodal:
        cmd.append("--use_multimodal")
    if args.sim3:
        cmd.append("--sim3")
    if args.se3:
        cmd.append("--se3")
    if args.no_ttt:
        cmd.append("--no_ttt")
    if args.no_swa:
        cmd.append("--no_swa")
    return cmd


def sanitize_log_name(relative_name: str) -> str:
    return relative_name.replace("/", "__")


def worker_loop(
    worker_name: str,
    gpu_id: str,
    args: argparse.Namespace,
    task_queue: "queue.Queue[ReconTask]",
    failures: List[str],
    failures_lock: threading.Lock,
) -> None:
    while True:
        try:
            task = task_queue.get_nowait()
        except queue.Empty:
            return

        start_time = time.time()
        cmd = build_command(args, task)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["OMP_NUM_THREADS"] = str(args.cpu_threads)
        env["MKL_NUM_THREADS"] = str(args.cpu_threads)
        env["OPENBLAS_NUM_THREADS"] = str(args.cpu_threads)
        env["NUMEXPR_NUM_THREADS"] = str(args.cpu_threads)
        env["PYTHONUNBUFFERED"] = "1"

        print(f"[{worker_name}] start gpu={gpu_id} task={task.relative_name}")

        log_handle = None
        try:
            if args.log_dir is not None:
                log_dir = Path(args.log_dir).expanduser().resolve()
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / f"{sanitize_log_name(task.relative_name)}.log"
                log_handle = open(log_path, "w", encoding="utf-8")
                result = subprocess.run(
                    cmd,
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            else:
                result = subprocess.run(cmd, env=env, text=True, check=False)
        finally:
            if log_handle is not None:
                log_handle.close()

        elapsed = round(time.time() - start_time, 2)
        if result.returncode != 0:
            message = f"[{worker_name}] failed gpu={gpu_id} task={task.relative_name} rc={result.returncode}"
            print(message)
            with failures_lock:
                failures.append(message)
        else:
            print(f"[{worker_name}] done gpu={gpu_id} task={task.relative_name} seconds={elapsed}")

        task_queue.task_done()


def main() -> None:
    args = parse_args()
    sample_root = Path(args.sample_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU id is required via --gpus")
    if args.workers_per_gpu <= 0:
        raise ValueError("--workers_per_gpu must be positive")
    if args.cpu_threads <= 0:
        raise ValueError("--cpu_threads must be positive")

    tasks = build_tasks(sample_root, output_root, force_annotation=args.force_annotation)
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]

    print(f"sample_root={sample_root}")
    print(f"output_root={output_root}")
    print(f"gpus={gpu_ids}")
    print(f"workers_per_gpu={args.workers_per_gpu}")
    print(f"cpu_threads={args.cpu_threads}")
    print(f"tasks={len(tasks)}")

    if not tasks:
        print("No reconstruction tasks found.")
        return

    if args.dry_run:
        for task in tasks[:20]:
            print(f"[dry_run] {task.relative_name} -> {task.output_dir}")
        if len(tasks) > 20:
            print(f"[dry_run] ... and {len(tasks) - 20} more tasks")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    task_queue: "queue.Queue[ReconTask]" = queue.Queue()
    for task in tasks:
        task_queue.put(task)

    failures: List[str] = []
    failures_lock = threading.Lock()
    workers: List[threading.Thread] = []

    worker_idx = 0
    for gpu_id in gpu_ids:
        for slot_idx in range(args.workers_per_gpu):
            name = f"worker-{worker_idx}-gpu{gpu_id}-slot{slot_idx}"
            thread = threading.Thread(
                target=worker_loop,
                args=(name, gpu_id, args, task_queue, failures, failures_lock),
                daemon=False,
                name=name,
            )
            workers.append(thread)
            worker_idx += 1

    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()

    if failures:
        print("\nFailed tasks:")
        for message in failures:
            print(message)
        raise SystemExit(1)

    print("All reconstruction tasks completed successfully.")


if __name__ == "__main__":
    main()
