from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch

from data_utils import get_meta_pt_strides, load_result_meta, load_result_tensors


@dataclass(frozen=True)
class OccupancyConfig:
    voxel_size: float
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    z_range: Tuple[float, float]
    point_stride: int
    temporal_radius: int
    conf_threshold: float
    occupancy_tau: float
    max_points_per_frame: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute sparse occupied voxels from reconstructed point clouds "
            "using a local occupancy voxel grid."
        )
    )
    parser.add_argument("--result_dir", type=str, required=True, help="A reconstruction result directory containing points.pt.")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory. Default: result_dir/collision_occupancy.")
    parser.add_argument("--voxel_size", type=float, default=0.25, help="Voxel edge length in meters.")
    parser.add_argument("--x_range", nargs=2, type=float, default=(-2.0, 2.0), metavar=("MIN", "MAX"), help="Collision ROI in current camera x axis.")
    parser.add_argument("--y_range", nargs=2, type=float, default=(-0.8, 0.8), metavar=("MIN", "MAX"), help="Collision ROI in current camera y axis.")
    parser.add_argument("--z_range", nargs=2, type=float, default=(0.2, 5.0), metavar=("MIN", "MAX"), help="Collision ROI in current camera forward z axis.")
    parser.add_argument("--point_stride", type=int, default=1, help="Spatial subsample stride on the stored point map.")
    parser.add_argument("--temporal_radius", type=int, default=0, help="Fuse +/- N neighboring stored point frames around each query frame.")
    parser.add_argument("--conf_threshold", type=float, default=0.3, help="Ignore points below this confidence. Values >1 are treated as percent.")
    parser.add_argument("--occupancy_tau", type=float, default=3.0, help="Weighted point count that maps to occupancy probability 1-exp(-1).")
    parser.add_argument("--max_points_per_frame", type=int, default=0, help="Random cap after valid filtering to keep runtime bounded. 0 disables the cap.")
    parser.add_argument("--save_frame_npz", action="store_true", help="Also save one sparse occupied-voxel npz per frame. Usually not needed.")
    parser.add_argument("--no_sparse_voxels", action="store_true", help="Do not save the compact sparse_occupancy.npz file.")
    parser.add_argument("--vis", action="store_true", help="Launch the same viser viewer used by vis_recon.py after computing outputs.")
    parser.add_argument("--frame_dir", type=str, default=None, help="RGB frame directory for --vis. If omitted, infer it by replacing /traj/ with /img/.")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame index for --vis.")
    parser.add_argument("--end_frame", type=int, default=-1, help="End frame index for --vis, exclusive. -1 means all frames.")
    parser.add_argument("--port", type=int, default=8090, help="Viser port used with --vis.")
    parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode.")
    parser.add_argument("--share", action="store_true", help="Request a public viser share URL.")
    parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation in the reused vis_recon viewer.")
    parser.add_argument("--vis_conf_threshold", type=float, default=20.0, help="Initial confidence threshold for the reused vis_recon viewer.")
    parser.add_argument("--subsample", type=int, default=5, help="Point cloud subsample factor for the reused vis_recon viewer.")
    parser.add_argument("--video_width", type=int, default=320, help="Video preview width for the reused vis_recon viewer.")
    parser.add_argument("--point_size", type=float, default=0.001, help="Point size for the reused vis_recon viewer.")
    parser.add_argument(
        "--reference_frame",
        type=str,
        default="auto",
        choices=["auto", "initial_camera", "result", "trajectory_plane"],
        help="Reference frame for the reused vis_recon viewer.",
    )
    parser.add_argument("--frame_step", type=int, default=8, help="Frame step used when reference_frame=trajectory_plane.")
    return parser.parse_args()


def _normalize_conf_threshold(value: float) -> float:
    value = float(value)
    if value > 1.0:
        value /= 100.0
    return float(np.clip(value, 0.0, 1.0))


def _normalize_pose_matrix(pose: torch.Tensor) -> torch.Tensor:
    if pose.shape == (4, 4):
        return pose
    if pose.shape == (3, 4):
        full = torch.eye(4, dtype=pose.dtype, device=pose.device)
        full[:3, :] = pose
        return full
    raise ValueError(f"Unexpected pose shape: {tuple(pose.shape)}")


def _grid_shape(config: OccupancyConfig) -> Tuple[int, int, int]:
    return (
        int(np.ceil((config.x_range[1] - config.x_range[0]) / config.voxel_size)),
        int(np.ceil((config.y_range[1] - config.y_range[0]) / config.voxel_size)),
        int(np.ceil((config.z_range[1] - config.z_range[0]) / config.voxel_size)),
    )


def _grid_dims(config: OccupancyConfig) -> Tuple[float, float, float]:
    return (
        float(config.x_range[1] - config.x_range[0]),
        float(config.y_range[1] - config.y_range[0]),
        float(config.z_range[1] - config.z_range[0]),
    )


def infer_frame_dir(result_dir: Path) -> Path:
    parts = list(result_dir.resolve().parts)
    if "traj" not in parts:
        raise ValueError("Could not infer frame_dir automatically because result_dir does not contain a 'traj' path component.")
    traj_idx = parts.index("traj")
    inferred_parts = parts[:]
    inferred_parts[traj_idx] = "img"
    return Path(*inferred_parts)


def _resolve_reference_frame(reference_frame: str, pred_dict: dict) -> str:
    if reference_frame != "auto":
        return reference_frame
    if pred_dict.get("use_result_frame", False):
        return "result"
    return "initial_camera"


def _select_pose_and_conf_for_points(
    tensors: Dict[str, torch.Tensor],
    meta: Dict[str, Any],
    point_frame_count: int,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    stride = get_meta_pt_strides(meta, 1)
    camera_poses = tensors["camera_poses"]
    if camera_poses.ndim == 4 and camera_poses.shape[0] == 1:
        camera_poses = camera_poses.squeeze(0)
    if int(camera_poses.shape[0]) != point_frame_count:
        camera_poses = camera_poses[::stride][:point_frame_count]
    camera_poses = torch.stack([_normalize_pose_matrix(pose) for pose in camera_poses], dim=0).to(torch.float32)

    conf = tensors.get("conf")
    if conf is None:
        return camera_poses, None
    if int(conf.shape[0]) != point_frame_count:
        conf = conf[::stride][:point_frame_count]
    return camera_poses, conf.to(torch.float32)


def _iter_temporal_indices(frame_idx: int, frame_count: int, radius: int) -> Iterable[int]:
    start = max(0, frame_idx - radius)
    end = min(frame_count, frame_idx + radius + 1)
    return range(start, end)


def _to_local_camera(points_world: torch.Tensor, camera_pose_c2w: torch.Tensor) -> torch.Tensor:
    world_to_camera = torch.linalg.inv(camera_pose_c2w.to(torch.float64)).to(torch.float32)
    rotation = world_to_camera[:3, :3]
    translation = world_to_camera[:3, 3]
    return points_world.to(torch.float32) @ rotation.T + translation


def _prepare_points_for_frame(
    points: torch.Tensor,
    conf: torch.Tensor | None,
    frame_idx: int,
    config: OccupancyConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    frame_points = points[frame_idx, :: config.point_stride, :: config.point_stride].reshape(-1, 3)
    valid = torch.isfinite(frame_points).all(dim=1)

    if conf is not None:
        frame_conf = conf[frame_idx, :: config.point_stride, :: config.point_stride].reshape(-1)
        valid = valid & torch.isfinite(frame_conf) & (frame_conf >= config.conf_threshold)
        weights = frame_conf[valid].to(torch.float32)
    else:
        weights = torch.ones(int(valid.sum().item()), dtype=torch.float32)

    frame_points = frame_points[valid]
    if config.max_points_per_frame > 0 and int(frame_points.shape[0]) > config.max_points_per_frame:
        indices = torch.randperm(int(frame_points.shape[0]))[: config.max_points_per_frame]
        frame_points = frame_points[indices]
        weights = weights[indices]
    return frame_points, weights


def compute_frame_occupancy(
    *,
    frame_idx: int,
    points: torch.Tensor,
    conf: torch.Tensor | None,
    camera_poses: torch.Tensor,
    config: OccupancyConfig,
) -> Dict[str, Any]:
    frame_count = int(points.shape[0])
    local_points_all = []
    weights_all = []
    pose = camera_poses[frame_idx]

    for source_idx in _iter_temporal_indices(frame_idx, frame_count, config.temporal_radius):
        source_points, source_weights = _prepare_points_for_frame(points, conf, source_idx, config)
        if source_points.numel() == 0:
            continue
        local_points_all.append(_to_local_camera(source_points, pose))
        weights_all.append(source_weights)

    if not local_points_all:
        return {
            "voxel_centers": np.empty((0, 3), dtype=np.float32),
            "voxel_indices": np.empty((0, 3), dtype=np.int32),
            "voxel_probabilities": np.empty((0,), dtype=np.float32),
            "occupied_voxels": 0,
        }

    local_points = torch.cat(local_points_all, dim=0)
    weights = torch.cat(weights_all, dim=0)
    xyz_min = torch.tensor([config.x_range[0], config.y_range[0], config.z_range[0]], dtype=torch.float32)
    xyz_max = torch.tensor([config.x_range[1], config.y_range[1], config.z_range[1]], dtype=torch.float32)
    in_roi = ((local_points >= xyz_min) & (local_points <= xyz_max)).all(dim=1)

    local_points = local_points[in_roi]
    weights = weights[in_roi]
    if local_points.numel() == 0:
        return {
            "voxel_centers": np.empty((0, 3), dtype=np.float32),
            "voxel_indices": np.empty((0, 3), dtype=np.int32),
            "voxel_probabilities": np.empty((0,), dtype=np.float32),
            "occupied_voxels": 0,
        }

    voxel_indices = torch.floor((local_points - xyz_min) / config.voxel_size).to(torch.int64)
    unique_indices, inverse = torch.unique(voxel_indices, dim=0, return_inverse=True)
    voxel_weights = torch.zeros(int(unique_indices.shape[0]), dtype=torch.float32)
    voxel_weights.scatter_add_(0, inverse.cpu(), weights.cpu())
    voxel_probabilities = 1.0 - torch.exp(-voxel_weights / max(config.occupancy_tau, 1e-6))
    voxel_probabilities = voxel_probabilities.clamp(0.0, 1.0)
    voxel_centers = xyz_min + (unique_indices.to(torch.float32) + 0.5) * config.voxel_size

    return {
        "voxel_centers": voxel_centers.numpy().astype(np.float32),
        "voxel_indices": unique_indices.numpy().astype(np.int32),
        "voxel_probabilities": voxel_probabilities.numpy().astype(np.float32),
        "occupied_voxels": int(voxel_probabilities.numel()),
    }


def run_collision_analysis(
    result_dir: Path,
    out_dir: Path,
    config: OccupancyConfig,
    *,
    save_frame_npz: bool,
    save_sparse_voxels: bool,
) -> Dict[str, Any]:
    meta = load_result_meta(result_dir)
    tensors = load_result_tensors(result_dir)
    points = tensors["points"]
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError(f"Expected points with shape [F,H,W,3], got {tuple(points.shape)}")
    camera_poses, conf = _select_pose_and_conf_for_points(tensors, meta, int(points.shape[0]))

    out_dir.mkdir(parents=True, exist_ok=True)
    frame_npz_dir = out_dir / "frames"
    if save_frame_npz:
        frame_npz_dir.mkdir(parents=True, exist_ok=True)

    sparse_frame_offsets = [0]
    sparse_voxel_indices = []
    sparse_voxel_probabilities = []
    start_time = time.time()
    frame_count = int(points.shape[0])
    for frame_idx in range(frame_count):
        result = compute_frame_occupancy(
            frame_idx=frame_idx,
            points=points,
            conf=conf,
            camera_poses=camera_poses,
            config=config,
        )
        sparse_voxel_indices.append(result["voxel_indices"])
        sparse_voxel_probabilities.append(result["voxel_probabilities"])
        sparse_frame_offsets.append(sparse_frame_offsets[-1] + int(result["voxel_indices"].shape[0]))
        if save_frame_npz:
            np.savez_compressed(
                frame_npz_dir / f"frame_{frame_idx:06d}.npz",
                voxel_indices=result["voxel_indices"],
                voxel_probabilities=result["voxel_probabilities"],
            )
        if (frame_idx + 1) % 25 == 0 or frame_idx + 1 == frame_count:
            print(f"processed {frame_idx + 1}/{frame_count} frames")

    sparse_path = out_dir / "sparse_occupancy.npz"
    metadata_path = out_dir / "metadata.json"
    total_sparse_voxels = int(sparse_frame_offsets[-1])
    if save_sparse_voxels:
        if sparse_voxel_indices:
            voxel_indices_np = np.concatenate(sparse_voxel_indices, axis=0).astype(np.int32, copy=False)
            voxel_probabilities_np = np.concatenate(sparse_voxel_probabilities, axis=0).astype(np.float32, copy=False)
        else:
            voxel_indices_np = np.empty((0, 3), dtype=np.int32)
            voxel_probabilities_np = np.empty((0,), dtype=np.float32)
        np.savez_compressed(
            sparse_path,
            frame_offsets=np.asarray(sparse_frame_offsets, dtype=np.int64),
            voxel_indices=voxel_indices_np,
            voxel_probabilities=voxel_probabilities_np,
        )
    summary = {
        "result_dir": str(result_dir),
        "out_dir": str(out_dir),
        "frames": frame_count,
        "grid_shape_xyz": list(_grid_shape(config)),
        "grid_dims_xyz_m": list(_grid_dims(config)),
        "grid_origin_xyz": [float(config.x_range[0]), float(config.y_range[0]), float(config.z_range[0])],
        "max_grid_voxels": int(np.prod(_grid_shape(config))),
        "total_sparse_voxels": total_sparse_voxels,
        "config": asdict(config),
        "elapsed_seconds": round(time.time() - start_time, 3),
        "files": {
            "sparse_occupancy": sparse_path.name if save_sparse_voxels else None,
            "metadata": metadata_path.name,
        },
    }
    metadata_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"summary": summary, "points": points, "conf": conf, "camera_poses": camera_poses, "config": config}


def _local_to_world_points(points_local: np.ndarray, pose_c2w: np.ndarray) -> np.ndarray:
    if points_local.size == 0:
        return points_local.astype(np.float32, copy=False)
    rotation = pose_c2w[:3, :3]
    translation = pose_c2w[:3, 3]
    return (points_local @ rotation.T + translation).astype(np.float32)


def _build_occupancy_overlay(pred_dict: dict, config: OccupancyConfig) -> dict:
    points = torch.from_numpy(np.asarray(pred_dict["points"])).to(torch.float32)
    conf = torch.from_numpy(np.asarray(pred_dict["conf"])).to(torch.float32) if "conf" in pred_dict else None
    camera_poses = torch.from_numpy(np.asarray(pred_dict["camera_poses"])).to(torch.float32)
    if camera_poses.ndim == 3:
        camera_poses = torch.stack([_normalize_pose_matrix(pose) for pose in camera_poses], dim=0)

    voxel_centers = []
    voxel_probabilities = []
    for frame_idx in range(int(points.shape[0])):
        occupancy = compute_frame_occupancy(
            frame_idx=frame_idx,
            points=points,
            conf=conf,
            camera_poses=camera_poses,
            config=config,
        )
        pose_np = camera_poses[frame_idx].cpu().numpy().astype(np.float32)
        voxel_centers.append(_local_to_world_points(occupancy["voxel_centers"], pose_np))
        voxel_probabilities.append(occupancy["voxel_probabilities"])

    return {
        "voxel_centers": voxel_centers,
        "voxel_probabilities": voxel_probabilities,
        "voxel_size": float(config.voxel_size),
        "grid_shape_xyz": list(_grid_shape(config)),
        "grid_dims_xyz_m": list(_grid_dims(config)),
    }


def launch_vis_recon_viewer(args: argparse.Namespace, result_dir: Path) -> None:
    from align_ground import apply_transform_to_points, apply_transform_to_poses, estimate_trajectory_frame
    from data_utils import load_result_for_viser
    from loger.utils.viser_utils import viser_wrapper

    frame_dir = Path(args.frame_dir).expanduser().resolve() if args.frame_dir is not None else infer_frame_dir(result_dir)
    pred_dict = load_result_for_viser(
        result_dir,
        frame_dir,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    pred_dict["sequence_name"] = result_dir.name
    reference_frame = _resolve_reference_frame(args.reference_frame, pred_dict)
    if reference_frame == "trajectory_plane":
        alignment = estimate_trajectory_frame(
            pred_dict["camera_poses"],
            frame_step=args.frame_step,
            plane_origin="first_camera",
        )
        pred_dict["points"] = apply_transform_to_points(pred_dict["points"], alignment["transform"])
        pred_dict["camera_poses"] = apply_transform_to_poses(pred_dict["camera_poses"], alignment["transform"])

    overlay_config = OccupancyConfig(
        voxel_size=max(float(args.voxel_size), 1e-6),
        x_range=(float(args.x_range[0]), float(args.x_range[1])),
        y_range=(float(args.y_range[0]), float(args.y_range[1])),
        z_range=(float(args.z_range[0]), float(args.z_range[1])),
        point_stride=max(1, int(args.point_stride)),
        temporal_radius=max(0, int(args.temporal_radius)),
        conf_threshold=_normalize_conf_threshold(args.conf_threshold),
        occupancy_tau=max(float(args.occupancy_tau), 1e-6),
        max_points_per_frame=max(0, int(args.max_points_per_frame)),
    )
    print("Building occupancy overlay for vis_recon viewer ...")
    pred_dict["collision_occupancy"] = _build_occupancy_overlay(pred_dict, overlay_config)

    canonical_first_frame = reference_frame == "initial_camera"
    viser_wrapper(
        pred_dict,
        port=args.port,
        init_conf_threshold=args.vis_conf_threshold,
        background_mode=args.background_mode,
        mask_sky=args.mask_sky,
        image_folder_for_sky_mask=str(frame_dir),
        subsample=args.subsample,
        video_width=args.video_width,
        share=args.share,
        point_size=args.point_size,
        canonical_first_frame=canonical_first_frame,
    )


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir is not None else result_dir / "collision_occupancy"
    config = OccupancyConfig(
        voxel_size=max(float(args.voxel_size), 1e-6),
        x_range=(float(args.x_range[0]), float(args.x_range[1])),
        y_range=(float(args.y_range[0]), float(args.y_range[1])),
        z_range=(float(args.z_range[0]), float(args.z_range[1])),
        point_stride=max(1, int(args.point_stride)),
        temporal_radius=max(0, int(args.temporal_radius)),
        conf_threshold=_normalize_conf_threshold(args.conf_threshold),
        occupancy_tau=max(float(args.occupancy_tau), 1e-6),
        max_points_per_frame=max(0, int(args.max_points_per_frame)),
    )
    analysis = run_collision_analysis(
        result_dir,
        out_dir,
        config,
        save_frame_npz=args.save_frame_npz,
        save_sparse_voxels=not args.no_sparse_voxels,
    )
    summary = analysis["summary"]
    print(json.dumps(summary, indent=2))

    if args.vis:
        print("Launching vis_recon viewer for the reconstruction result.")
        launch_vis_recon_viewer(args, result_dir)


if __name__ == "__main__":
    main()
