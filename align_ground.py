from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data_utils import (
    list_image_files,
    load_result_meta,
    load_result_tensors,
    save_result_directory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate a trajectory plane from camera poses and save a rotated result in that global frame."
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
        help="How to place the world origin on the fitted motion plane.",
    )
    parser.add_argument(
        "--frame_step",
        type=int,
        default=1,
        help="Use every Nth camera center when fitting the trajectory plane.",
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


def _fit_plane_to_camera_centers(camera_centers: np.ndarray, frame_step: int) -> tuple[np.ndarray, np.ndarray]:
    sampled = camera_centers[:: max(1, int(frame_step))]
    if len(sampled) < 3:
        sampled = camera_centers
    if len(sampled) < 3:
        raise RuntimeError("Need at least three camera poses to estimate a trajectory plane.")

    plane_point = sampled.mean(axis=0)
    centered = sampled - plane_point
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    plane_normal = _normalize(vh[-1])
    return plane_point, plane_normal


def _estimate_up_axis(camera_centers: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    signed_distances = (camera_centers - plane_point) @ plane_normal
    up_axis = plane_normal if float(np.median(signed_distances)) >= 0.0 else -plane_normal
    up_axis = _normalize(up_axis)
    if up_axis[1] < 0.0:
        up_axis = -up_axis
    return up_axis


def _estimate_forward_axis(camera_centers: np.ndarray, up_axis: np.ndarray) -> np.ndarray:
    displacements = np.diff(camera_centers, axis=0)
    displacements = displacements - np.outer(displacements @ up_axis, up_axis)
    valid = np.linalg.norm(displacements, axis=1) > 1e-6
    if np.any(valid):
        forward_axis = displacements[valid].sum(axis=0)
    else:
        forward_axis = camera_centers[-1] - camera_centers[0]
        forward_axis = forward_axis - np.dot(forward_axis, up_axis) * up_axis
    if np.linalg.norm(forward_axis) < 1e-8:
        raise RuntimeError("Could not estimate a forward direction from the trajectory.")
    forward_axis = _normalize(forward_axis)

    global_direction = camera_centers[-1] - camera_centers[0]
    global_direction = global_direction - np.dot(global_direction, up_axis) * up_axis
    if np.linalg.norm(global_direction) > 1e-8 and np.dot(forward_axis, global_direction) < 0:
        forward_axis = -forward_axis
    return forward_axis


def estimate_trajectory_frame(
    camera_poses: torch.Tensor | np.ndarray,
    *,
    frame_step: int = 1,
    plane_origin: str = "first_camera",
) -> dict[str, np.ndarray]:
    camera_centers = camera_centers_from_poses(camera_poses)
    plane_point, plane_normal = _fit_plane_to_camera_centers(camera_centers, frame_step=frame_step)
    up_axis = _estimate_up_axis(camera_centers, plane_point, plane_normal)
    forward_axis = _estimate_forward_axis(camera_centers, up_axis)
    right_axis = _normalize(np.cross(up_axis, forward_axis))
    forward_axis = _normalize(np.cross(right_axis, up_axis))

    if plane_origin == "centroid":
        origin = plane_point
    elif plane_origin == "first_camera":
        first_camera = camera_centers[0]
        origin = first_camera - np.dot(first_camera - plane_point, up_axis) * up_axis
    else:
        raise ValueError(f"Unsupported plane_origin: {plane_origin}")

    rotation = np.stack([right_axis, up_axis, forward_axis], axis=0)
    translation = -rotation @ origin

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation

    signed_distances = (camera_centers - plane_point) @ up_axis
    return {
        "transform": transform,
        "rotation": rotation,
        "translation": translation,
        "plane_point": plane_point,
        "plane_normal": up_axis,
        "up_axis": up_axis,
        "forward_axis": forward_axis,
        "right_axis": right_axis,
        "origin": origin,
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
        input_frame_stride=int(meta.get("stride", 1)),
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
    print(f"Median camera height above plane: {float(alignment['camera_height_median'][0]):.4f}")
    print(f"Saved alignment file: {alignment_path}")
    print(f"Saved aligned meta: {result_meta}")


if __name__ == "__main__":
    main()
