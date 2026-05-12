from __future__ import annotations

import argparse
import contextlib
import multiprocessing as mp
import os
import queue
import time
import traceback
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
    output_dir: Path
    relative_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel reconstruction launcher that keeps one model loaded per worker."
    )
    parser.add_argument("--sample_root", type=str, required=True, help="Root img directory, e.g. /data/xby/YTB/shanghai0/img")
    parser.add_argument("--output_root", type=str, required=True, help="Root traj directory, e.g. /data/xby/YTB/shanghai0/traj")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--workers_per_gpu", type=int, default=1, help="Concurrent reconstruction workers per GPU.")
    parser.add_argument("--cpu_threads", type=int, default=4, help="OMP/MKL/OpenBLAS threads per worker process.")
    parser.add_argument("--decode_workers", type=int, default=None, help="Threads used to decode images inside each worker.")
    parser.add_argument("--max_tasks", type=int, default=None, help="Optional limit on the number of leaf frame folders.")
    parser.add_argument("--log_dir", type=str, default=None, help="Optional directory for one log file per task.")
    parser.add_argument("--dry_run", action="store_true", help="Print planned tasks without launching reconstruction.")

    parser.add_argument("--model_name", type=str, default="ckpts/Pi3X", help="Local HF Pi3X dir or local LoGeR checkpoint.")
    parser.add_argument("--config", type=str, default="ckpts/LoGeR_star/original_config.yaml", help="LoGeR config path.")
    parser.add_argument("--window_size", type=int, default=32, help="Window size for chunked inference.")
    parser.add_argument("--overlap_size", type=int, default=3, help="Overlap size between windows.")
    parser.add_argument("--window_batch_size", type=int, default=4, help="Number of windows to run in one Pi3X forward pass.")
    parser.add_argument("--reset_every", type=int, default=None, help="Reset interval for merge semantics.")
    parser.add_argument("--stride", type=int, default=3, help="Save every Nth frame result.")
    parser.add_argument("--conf_threshold", type=float, default=30.0, help="Confidence threshold for exported results.")
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
    parser.add_argument("--se3", action="store_true", default=None, help="Use SE3 alignment.")
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

        tasks.append(
            ReconTask(
                frame_dir=frame_dir,
                output_dir=output_dir,
                relative_name=relative_path.as_posix(),
            )
        )
    return tasks


def sanitize_log_name(relative_name: str) -> str:
    return relative_name.replace("/", "__")


def configure_worker_env(gpu_id: str, cpu_threads: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_threads)
    os.environ["PYTHONUNBUFFERED"] = "1"


def process_task(
    *,
    task: ReconTask,
    args: argparse.Namespace,
    device,
    model,
    model_kind: str,
    forward_kwargs,
    target_resolution,
    list_image_files,
    load_images_from_paths,
    save_result_directory,
    run_inference,
) -> None:
    image_paths = list_image_files(task.frame_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {task.frame_dir}")

    images_tensor = load_images_from_paths(
        image_paths,
        target_resolution=target_resolution,
        verbose=False,
        device=device,
        decode_workers=args.decode_workers,
    )
    predictions, stats = run_inference(
        model,
        model_kind,
        images_tensor,
        device=device,
        forward_kwargs=forward_kwargs,
    )
    save_result_directory(
        task.output_dir,
        predictions,
        frame_dir=task.frame_dir,
        image_paths=image_paths,
        model_name=args.model_name,
        model_kind=model_kind,
        target_resolution=target_resolution,
        forward_kwargs=forward_kwargs,
        stride=args.stride,
        conf_threshold=args.conf_threshold,
        inference_stats=stats,
        overwrite=True,
    )


def worker_main(
    worker_name: str,
    gpu_id: str,
    args: argparse.Namespace,
    task_queue: "mp.Queue[ReconTask | None]",
    result_queue: "mp.Queue[tuple[str, str, str, float]]",
) -> None:
    configure_worker_env(gpu_id, args.cpu_threads)

    import torch

    from data_utils import save_result_directory
    from loger.reconstruction import (
        build_forward_kwargs,
        load_reconstruction_model,
        run_inference,
    )
    from tensor_utils import list_image_files, load_images_from_paths

    try:
        torch.set_num_threads(args.cpu_threads)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(max(1, min(4, args.cpu_threads)))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(0)
        target_resolution = tuple(args.resolution) if args.resolution is not None else None

        print(f"[{worker_name}] device={device} visible_gpu={gpu_id}")
        print(f"[{worker_name}] Loading model once from {args.model_name}")
        model, model_kind = load_reconstruction_model(
            args.model_name,
            model_config_path=args.config,
            pi3x=True,
            pi3x_metric=True,
            use_multimodal=args.use_multimodal,
            device=device,
        )
        forward_kwargs = build_forward_kwargs(
            config_path=args.config,
            window_size=args.window_size,
            overlap_size=args.overlap_size,
            window_batch_size=args.window_batch_size,
            reset_every=args.reset_every,
            sim3=args.sim3,
            se3=args.se3,
            sim3_scale_mode=args.sim3_scale_mode,
            no_ttt=args.no_ttt,
            no_swa=args.no_swa,
        )
        forward_kwargs["window_batch_size"] = args.window_batch_size
        print(f"[{worker_name}] model ready kind={model_kind} forward_kwargs={forward_kwargs}")
    except Exception:
        result_queue.put(("worker_fail", worker_name, traceback.format_exc(), 0.0))
        raise

    while True:
        task = task_queue.get()
        if task is None:
            print(f"[{worker_name}] no more tasks, exiting")
            return

        start_time = time.time()
        print(f"[{worker_name}] start gpu={gpu_id} task={task.relative_name}")
        try:
            log_handle = None
            if args.log_dir is not None:
                log_dir = Path(args.log_dir).expanduser().resolve()
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / f"{sanitize_log_name(task.relative_name)}.log"
                log_handle = open(log_path, "w", encoding="utf-8")

            with contextlib.ExitStack() as stack:
                if log_handle is not None:
                    stack.enter_context(log_handle)
                    stack.enter_context(contextlib.redirect_stdout(log_handle))
                    stack.enter_context(contextlib.redirect_stderr(log_handle))
                    print(f"[{worker_name}] start gpu={gpu_id} task={task.relative_name}")

                process_task(
                    task=task,
                    args=args,
                    device=device,
                    model=model,
                    model_kind=model_kind,
                    forward_kwargs=forward_kwargs,
                    target_resolution=target_resolution,
                    list_image_files=list_image_files,
                    load_images_from_paths=load_images_from_paths,
                    save_result_directory=save_result_directory,
                    run_inference=run_inference,
                )

            elapsed = round(time.time() - start_time, 2)
            print(f"[{worker_name}] done gpu={gpu_id} task={task.relative_name} seconds={elapsed}")
            result_queue.put(("ok", task.relative_name, "", elapsed))
        except Exception:
            elapsed = round(time.time() - start_time, 2)
            error_text = traceback.format_exc()
            print(f"[{worker_name}] failed gpu={gpu_id} task={task.relative_name} seconds={elapsed}")
            print(error_text)
            result_queue.put(("fail", task.relative_name, error_text, elapsed))
        finally:
            pass


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

    total_workers = len(gpu_ids) * args.workers_per_gpu

    print(f"sample_root={sample_root}")
    print(f"output_root={output_root}")
    print(f"gpus={gpu_ids}")
    print(f"workers_per_gpu={args.workers_per_gpu}")
    print(f"total_workers={total_workers}")
    print(f"cpu_threads={args.cpu_threads}")
    print(f"tasks={len(tasks)}")
    if args.workers_per_gpu > 1:
        print("note=each worker keeps its own model copy, so repeated loads per GPU will equal workers_per_gpu")

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

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    for task in tasks:
        task_queue.put(task)
    for _ in range(total_workers):
        task_queue.put(None)

    workers: List[mp.Process] = []
    worker_idx = 0
    for gpu_id in gpu_ids:
        for slot_idx in range(args.workers_per_gpu):
            name = f"worker-{worker_idx}-gpu{gpu_id}-slot{slot_idx}"
            process = ctx.Process(
                target=worker_main,
                args=(name, gpu_id, args, task_queue, result_queue),
                name=name,
                daemon=False,
            )
            workers.append(process)
            worker_idx += 1

    for process in workers:
        process.start()

    for process in workers:
        process.join()

    successes = 0
    failures: List[tuple[str, str]] = []
    worker_failures: List[tuple[str, str]] = []
    while True:
        try:
            status, name, message, _elapsed = result_queue.get_nowait()
        except queue.Empty:
            break
        if status == "ok":
            successes += 1
        elif status == "fail":
            failures.append((name, message))
        elif status == "worker_fail":
            worker_failures.append((name, message))

    exit_failures = [process for process in workers if process.exitcode not in (0, None)]
    if failures or worker_failures or exit_failures:
        if failures:
            print("\nFailed tasks:")
            for name, message in failures:
                print(f"- {name}")
                print(message)
        if worker_failures:
            print("\nWorker startup failures:")
            for name, message in worker_failures:
                print(f"- {name}")
                print(message)
        if exit_failures:
            print("\nWorker exit codes:")
            for process in exit_failures:
                print(f"- {process.name}: exitcode={process.exitcode}")
        raise SystemExit(1)

    print(f"All reconstruction tasks completed successfully. completed={successes}")


if __name__ == "__main__":
    main()
