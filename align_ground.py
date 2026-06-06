from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data_utils import (
    get_meta_pt_strides,
    list_image_files,
    load_result_meta,
    load_result_tensors,
    save_result_directory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate a stable trajectory-aligned global frame from camera poses and save a rotated result in that frame."
    )
    parser.add_argument("--result_dir", type=str, required=True, help="Input reconstruction result directory.")
    parser.add_argument("--frame_dir", type=str, required=True, help="RGB frame directory corresponding to the result.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for the aligned result. Default: <result_dir>_trajplane",
    )
    parser.add_argument(
        "--plane_origin",
        type=str,
        default="first_camera",
        choices=["first_camera", "centroid"],
        help="How to place the world origin in the estimated trajectory-aligned frame.",
    )
    parser.add_argument(
        "--frame_step",
        type=int,
        default=1,
        help="Use the first N frames to estimate the up axis, and use the displacement from frame 0 to frame N to estimate the forward axis.",
    )
    parser.add_argument(
        "--camera_y_axis",
        type=str,
        default="down",
        choices=["down", "up"],
        help="Interpret the camera local y-axis as pointing down or up when estimating the global up direction.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output directory.")
    return parser.parse_args()


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise RuntimeError("Encountered a near-zero vector while estimating the trajectory frame.")
    return vector / norm


def camera_centers_from_poses(camera_poses: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(camera_poses):
        poses = camera_poses.detach().cpu().float().numpy()
    else:
        poses = np.asarray(camera_poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected camera_poses with shape [N, 4, 4], got {tuple(poses.shape)}")
    return poses[:, :3, 3]


def _camera_rotations_from_poses(camera_poses: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(camera_poses):
        poses = camera_poses.detach().cpu().float().numpy()
    else:
        poses = np.asarray(camera_poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected camera_poses with shape [N, 4, 4], got {tuple(poses.shape)}")
    return poses[:, :3, :3]


def _estimate_up_axis_from_initial_rotations(
    camera_poses: torch.Tensor | np.ndarray,
    *,
    frame_step: int,
    camera_y_axis: str,
    max_outlier_angle_deg: float = 35.0,
) -> np.ndarray:
    rotations = _camera_rotations_from_poses(camera_poses)
    if len(rotations) == 0:
        raise RuntimeError("Need at least one camera pose to estimate the global up axis.")

    frame_count = max(1, min(int(frame_step), len(rotations)))
    local_up_sign = -1.0 if camera_y_axis == "down" else 1.0
    up_vectors = local_up_sign * rotations[:frame_count, :, 1]
    up_vectors = np.stack([_normalize(vector) for vector in up_vectors], axis=0)

    reference = up_vectors[0]
    aligned = up_vectors.copy()
    flip_mask = (aligned @ reference) < 0.0
    aligned[flip_mask] *= -1.0

    mean_up = _normalize(aligned.mean(axis=0))
    cos_threshold = float(np.cos(np.deg2rad(max_outlier_angle_deg)))
    keep_mask = (aligned @ mean_up) >= cos_threshold
    if np.any(keep_mask):
        mean_up = _normalize(aligned[keep_mask].mean(axis=0))

    if mean_up[1] < 0.0:
        mean_up = -mean_up
    return mean_up


def _estimate_forward_axis_from_initial_displacement(
    camera_poses: torch.Tensor | np.ndarray,
    up_axis: np.ndarray,
    *,
    frame_step: int,
) -> np.ndarray:
    camera_centers = camera_centers_from_poses(camera_poses)
    if len(camera_centers) < 2:
        raise RuntimeError("Need at least one camera pose to estimate a forward direction.")
    step = max(1, int(frame_step))
    candidate_indices = [min(step, len(camera_centers) - 1)] + list(range(step + 1, len(camera_centers)))
    tried = set()
    for idx in candidate_indices:
        if idx <= 0 or idx in tried:
            continue
        tried.add(idx)
        forward_axis = camera_centers[idx] - camera_centers[0]
        forward_axis = forward_axis - np.dot(forward_axis, up_axis) * up_axis
        if np.linalg.norm(forward_axis) >= 1e-8:
            return _normalize(forward_axis)
    raise RuntimeError("The initial translation is too small or nearly parallel to the estimated up axis, so a stable forward direction could not be defined.")


def estimate_trajectory_frame(
    camera_poses: torch.Tensor | np.ndarray,
    *,
    frame_step: int = 1,
    plane_origin: str = "first_camera",
    camera_y_axis: str = "down",
) -> dict[str, np.ndarray]:
    camera_centers = camera_centers_from_poses(camera_poses)
    sampled_centers = camera_centers[:: max(1, int(frame_step))]
    if len(sampled_centers) < 2:
        sampled_centers = camera_centers

    up_axis = _estimate_up_axis_from_initial_rotations(
        camera_poses,
        frame_step=frame_step,
        camera_y_axis=camera_y_axis,
    )
    forward_axis = _estimate_forward_axis_from_initial_displacement(
        camera_poses,
        up_axis,
        frame_step=frame_step,
    )
    right_axis = _normalize(np.cross(up_axis, forward_axis))
    forward_axis = _normalize(np.cross(right_axis, up_axis))
    trajectory_centroid = sampled_centers.mean(axis=0)

    if plane_origin == "centroid":
        origin = trajectory_centroid
    elif plane_origin == "first_camera":
        origin = camera_centers[0]
    else:
        raise ValueError(f"Unsupported plane_origin: {plane_origin}")

    rotation = np.stack([right_axis, up_axis, forward_axis], axis=0)
    translation = -rotation @ origin

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation

    signed_distances = (camera_centers - origin) @ up_axis
    return {
        "transform": transform,
        "rotation": rotation,
        "translation": translation,
        "plane_point": trajectory_centroid,
        "plane_normal": up_axis,
        "up_axis": up_axis,
        "forward_axis": forward_axis,
        "right_axis": right_axis,
        "origin": origin,
        "frame_step_used": np.array([min(max(1, int(frame_step)), len(camera_centers))], dtype=np.int64),
        "camera_y_axis_sign": np.array([-1.0 if camera_y_axis == "down" else 1.0], dtype=np.float64),
        "camera_height_median": np.array([float(np.median(signed_distances))], dtype=np.float64),
        "camera_height_std": np.array([float(np.std(signed_distances))], dtype=np.float64),
    }


def apply_transform_to_points(points: torch.Tensor | np.ndarray, transform: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    if torch.is_tensor(points):
        transform_t = transform if torch.is_tensor(transform) else torch.from_numpy(np.asarray(transform))
        rotation = transform_t[:3, :3].to(device=points.device, dtype=torch.float32)
        translation = transform_t[:3, 3].to(device=points.device, dtype=torch.float32)
        result = points.detach().clone().to(torch.float32)
        valid = torch.isfinite(result).all(dim=-1)
        if torch.any(valid):
            result[valid] = result[valid] @ rotation.T + translation
        return result

    points_np = np.asarray(points, dtype=np.float32).copy()
    transform_np = transform.detach().cpu().numpy() if torch.is_tensor(transform) else np.asarray(transform, dtype=np.float64)
    rotation = transform_np[:3, :3].astype(np.float32, copy=False)
    translation = transform_np[:3, 3].astype(np.float32, copy=False)
    valid = np.isfinite(points_np).all(axis=-1)
    if np.any(valid):
        points_np[valid] = points_np[valid] @ rotation.T + translation
    return points_np


def apply_transform_to_poses(camera_poses: torch.Tensor | np.ndarray, transform: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    if torch.is_tensor(camera_poses):
        transform_t = transform if torch.is_tensor(transform) else torch.from_numpy(np.asarray(transform))
        return torch.einsum("ij,njk->nik", transform_t.to(device=camera_poses.device, dtype=torch.float32), camera_poses.to(torch.float32))

    poses_np = np.asarray(camera_poses, dtype=np.float32)
    transform_np = transform.detach().cpu().numpy() if torch.is_tensor(transform) else np.asarray(transform, dtype=np.float32)
    return np.einsum("ij,njk->nik", transform_np, poses_np)


def _save_alignment_file(output_dir: Path, alignment: dict[str, np.ndarray]) -> Path:
    output_path = output_dir / "trajectory_plane_alignment.npz"
    np.savez_compressed(output_path, **alignment)
    return output_path


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir).expanduser().resolve()
    frame_dir = Path(args.frame_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else result_dir.parent / f"{result_dir.name}_trajplane"
    )
    if output_dir.exists() and not args.force:
        raise FileExistsError(f"Output directory already exists: {output_dir}. Use --force to overwrite it.")

    meta = load_result_meta(result_dir)
    tensors = load_result_tensors(result_dir)
    image_paths = list_image_files(frame_dir)

    alignment = estimate_trajectory_frame(
        tensors["camera_poses"],
        frame_step=args.frame_step,
        plane_origin=args.plane_origin,
        camera_y_axis=args.camera_y_axis,
    )
    transform = torch.from_numpy(alignment["transform"]).to(dtype=torch.float32)

    predictions = {
        "points": apply_transform_to_points(tensors["points"], transform),
        "conf": tensors["conf"],
        "camera_poses": apply_transform_to_poses(tensors["camera_poses"], transform),
        "depth_maps": tensors["depth_maps"],
    }

    target_resolution = tuple(meta["target_resolution"]) if meta.get("target_resolution") is not None else None
    result_meta = save_result_directory(
        output_dir,
        predictions,
        frame_dir=frame_dir,
        image_paths=image_paths,
        model_name=str(meta.get("model_name", "unknown")),
        model_kind=str(meta.get("model_kind", "unknown")),
        target_resolution=target_resolution,
        forward_kwargs={
            "window_size": int(meta.get("window_size", 32)),
            "overlap_size": int(meta.get("overlap_size", 3)),
        },
        stride=1,
        input_frame_stride=get_meta_pt_strides(meta, 1),
        conf_threshold=float(meta.get("conf_threshold", 0.0)),
        inference_stats=dict(meta.get("inference_stats", {})),
        canonical_first_frame_for_plot=False,
        extra_meta={
            "reference_frame": "trajectory_plane",
            "alignment_file": "trajectory_plane_alignment.npz",
            "trajectory_plane": {
                "plane_point": alignment["plane_point"].tolist(),
                "plane_normal": alignment["plane_normal"].tolist(),
                "up_axis": alignment["up_axis"].tolist(),
                "forward_axis": alignment["forward_axis"].tolist(),
                "right_axis": alignment["right_axis"].tolist(),
                "origin": alignment["origin"].tolist(),
                "frame_step_used": int(alignment["frame_step_used"][0]),
                "camera_y_axis": args.camera_y_axis,
                "camera_height_median": float(alignment["camera_height_median"][0]),
                "camera_height_std": float(alignment["camera_height_std"][0]),
                "plane_origin_mode": args.plane_origin,
            },
        },
        overwrite=args.force,
    )
    alignment_path = _save_alignment_file(output_dir, alignment)

    print(f"Input result: {result_dir}")
    print(f"Frame directory: {frame_dir}")
    print(f"Output result: {output_dir}")
    print(f"Estimated up axis: {alignment['up_axis']}")
    print(f"Estimated forward axis: {alignment['forward_axis']}")
    print(f"Origin mode: {args.plane_origin}")
    print(f"Frame step used for up/forward estimation: {int(alignment['frame_step_used'][0])}")
    print(f"Median camera height above reference horizontal plane: {float(alignment['camera_height_median'][0]):.4f}")
    print(f"Saved alignment file: {alignment_path}")
    print(f"Saved aligned meta: {result_meta}")


if __name__ == "__main__":
    main()
