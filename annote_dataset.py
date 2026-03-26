import argparse
from pathlib import Path

import torch

from data_utils import list_image_files, load_images_from_paths, save_result_directory
from loger.reconstruction import (
    build_forward_kwargs,
    load_reconstruction_model,
    run_inference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct sampled image folders with Pi3X + LoGeR-style windowing.")
    parser.add_argument("--sampled_root", type=str, default="data", help="Directory containing sampled image folders.")
    parser.add_argument("--output_root", type=str, default="results", help="Directory to store result folders.")
    parser.add_argument("--model_name", type=str, default="ckpts/Pi3X", help="Local HF Pi3X dir or LoGeR/Pi3 checkpoint.")
    parser.add_argument("--config", type=str, default="ckpts/LoGeR_star/original_config.yaml", help="LoGeR config used to inherit window/merge defaults.")

    parser.add_argument("--pi3x", action="store_true", default=True, help="Use Pi3X model.")
    parser.add_argument("--pi3x_metric", action="store_true", default=True, help="Use metric scaling for Pi3X.")
    parser.add_argument("--use_multimodal", action="store_true", help="Enable Pi3X multimodal branch. Disabled by default for RGB-only videos.")

    parser.add_argument("--window_size", type=int, default=32, help="Window size for chunked inference.")
    parser.add_argument("--overlap_size", type=int, default=3, help="Overlap size between windows.")
    parser.add_argument("--reset_every", type=int, default=None, help="Reset interval used by LoGeR merge semantics.")
    parser.add_argument("--sim3", action="store_true", help="Use Sim3 alignment when merging windows.")
    parser.add_argument("--se3", action="store_true", default=None, help="Use SE3 alignment when merging windows.")
    parser.add_argument("--sim3_scale_mode", type=str, default="median", choices=["median", "trimmed_mean", "median_all", "trimmed_mean_all", "sim3_avg1"], help="Scale estimation mode for Sim3 merge.")
    
    parser.add_argument("--no_ttt", action="store_true", help="Forwarded for LoGeR Pi3 checkpoints.")
    parser.add_argument("--no_swa", action="store_true", help="Forwarded for LoGeR Pi3 checkpoints.")
    
    parser.add_argument("--force_annotation", action="store_true", help="Overwrite existing annotation outputs.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional limit on the number of sampled folders to process.")
    parser.add_argument(
        "--resolution",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=(672,378),
        help="Target resolution override.",
    )
    return parser.parse_args()


def _list_sampled_folders(sampled_root: Path) -> list[Path]:
    folders = []
    for path in sorted(sampled_root.iterdir()):
        if path.is_dir() and list_image_files(path):
            folders.append(path)
    return folders


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_resolution = tuple(args.resolution) if args.resolution is not None else None

    sampled_root = Path(args.sampled_root)
    output_root = Path(args.output_root)
    folders = _list_sampled_folders(sampled_root)
    if args.max_samples is not None:
        folders = folders[: args.max_samples]
    if not folders:
        raise FileNotFoundError(f"No sampled image folders found in {sampled_root}")

    print(f"Using device: {device}")
    print(f"Loading model once from {args.model_name}")
    model, model_kind = load_reconstruction_model(
        args.model_name,
        model_config_path=args.config,
        pi3x=args.pi3x,
        pi3x_metric=args.pi3x_metric,
        use_multimodal=args.use_multimodal,
        device=device,
    )

    forward_kwargs = build_forward_kwargs(
        config_path=args.config,
        window_size=args.window_size,
        overlap_size=args.overlap_size,
        reset_every=args.reset_every,
        sim3=args.sim3,
        se3=args.se3,
        sim3_scale_mode=args.sim3_scale_mode,
        no_ttt=args.no_ttt,
        no_swa=args.no_swa,
    )
    print(f"Forward/merge kwargs: {forward_kwargs}")

    for index, frame_dir in enumerate(folders, start=1):
        image_paths = list_image_files(frame_dir)
        if not image_paths:
            continue

        print(f"\n[{index}/{len(folders)}] Processing {frame_dir.name}")
        output_dir = output_root / frame_dir.name
        if output_dir.exists() and not args.force_annotation:
            required = ["meta.yaml", "camera_poses.pt", "depth_maps.pt", "points.pt", "conf.pt", "trajectory_xz.png"]
            if all((output_dir / name).exists() for name in required):
                print(f"  Skipping annotation because output already exists: {output_dir}")
                continue

        images_tensor = load_images_from_paths(
            image_paths,
            target_resolution=target_resolution,
            verbose=False,
        )
        predictions, stats = run_inference(
            model,
            model_kind,
            images_tensor,
            device=device,
            forward_kwargs=forward_kwargs,
        )
        meta = save_result_directory(
            output_dir,
            predictions,
            frame_dir=frame_dir,
            image_paths=image_paths,
            model_name=args.model_name,
            model_kind=model_kind,
            target_resolution=target_resolution,
            forward_kwargs=forward_kwargs,
            source_video=None,
            inference_stats=stats,
            overwrite=True,
        )
        print(f"  Saved result directory to {output_dir}")
        print(f"  Trajectory plot: {meta['trajectory_plot']}")


if __name__ == "__main__":
    main()
