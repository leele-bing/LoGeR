from __future__ import annotations

import glob
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from natsort import natsorted
from torchvision import transforms

from loger.utils.geometry import homogenize_points


def list_image_files(input_dir: str | Path) -> List[str]:
    paths: List[str] = []
    base_dir = Path(input_dir)
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        paths.extend(glob.glob(str(base_dir / pattern)))
    image_paths = natsorted(paths)
    return [path for path in image_paths if "depth" not in Path(path).name.lower()]


def load_images_from_paths(
    image_paths: Sequence[str],
    *,
    pixel_limit: int = 255000,
    target_resolution: Optional[Tuple[int, int]] = None,
    verbose: bool = True,
) -> torch.Tensor:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    if not images:
        return torch.empty(0)

    if target_resolution is None:
        width, height = images[0].size
        scale = math.sqrt(pixel_limit / (width * height)) if width * height > 0 else 1.0
        scaled_width = width * scale
        scaled_height = height * scale
        width_units = round(scaled_width / 14)
        height_units = round(scaled_height / 14)
        while (width_units * 14) * (height_units * 14) > pixel_limit:
            if width_units / max(height_units, 1) > scaled_width / max(scaled_height, 1):
                width_units -= 1
            else:
                height_units -= 1
        target_width = max(1, width_units) * 14
        target_height = max(1, height_units) * 14
    else:
        target_width, target_height = target_resolution

    if verbose:
        print(f"All images will be resized to a uniform size: ({target_width}, {target_height})")

    to_tensor = transforms.ToTensor()
    resized_tensors = []
    for image in images:
        resized = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        resized_tensors.append(to_tensor(resized))
    return torch.stack(resized_tensors, dim=0)


def _normalize_pose_matrix(pose: torch.Tensor) -> torch.Tensor:
    if pose.shape == (4, 4):
        return pose
    if pose.shape == (3, 4):
        full = torch.eye(4, dtype=pose.dtype, device=pose.device)
        full[:3, :] = pose
        return full
    raise ValueError(f"Unexpected pose shape: {tuple(pose.shape)}")


def _camera_centers_from_poses(camera_poses: torch.Tensor) -> torch.Tensor:
    poses = camera_poses
    if poses.ndim == 4 and poses.shape[0] == 1:
        poses = poses.squeeze(0)
    if poses.ndim != 3:
        raise ValueError(f"Expected camera_poses with shape [N, 3/4, 4/4], got {tuple(poses.shape)}")
    normalized = torch.stack([_normalize_pose_matrix(pose) for pose in poses], dim=0)
    return normalized[:, :3, 3]


def _save_trajectory_xz_plot(output_path: Path, camera_poses: torch.Tensor) -> Path:
    centers = _camera_centers_from_poses(camera_poses).detach().cpu().float().numpy()
    if centers.size == 0:
        raise RuntimeError("Cannot plot trajectory without camera centers.")

    x = centers[:, 0]
    z = centers[:, 2]
    colors = plt.get_cmap("viridis")(np.linspace(0.0, 1.0, len(centers)))

    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    ax.plot(x, z, color="0.35", linewidth=1.5, alpha=0.8)
    ax.scatter(x, z, c=colors, s=14, linewidths=0)
    ax.scatter([x[0]], [z[0]], c=["#2ca02c"], s=48, label="start", zorder=3)
    ax.scatter([x[-1]], [z[-1]], c=["#d62728"], s=48, label="end", zorder=3)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("Camera Trajectory on XZ Plane")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best")
    ax.text(0.02, 0.02, f"frames: {len(centers)}", transform=ax.transAxes, fontsize=9, color="0.35", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _resolve_target_resolution(
    target_resolution: Optional[Tuple[int, int]],
    points: torch.Tensor,
) -> Tuple[int, int]:
    if target_resolution is not None:
        return int(target_resolution[0]), int(target_resolution[1])
    if points.ndim < 3:
        raise RuntimeError("Could not infer target resolution from points tensor.")
    return int(points.shape[2]), int(points.shape[1])


def _resolve_result_file(result_dir: Path, meta: Dict[str, Any], key: str, filename: str) -> Path:
    candidate = (meta.get("files") or {}).get(key)
    if candidate:
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return candidate_path
    fallback = result_dir / filename
    if fallback.exists():
        return fallback
    if candidate:
        raise FileNotFoundError(f"Could not find {key} at {candidate} or {fallback}")
    raise FileNotFoundError(f"Could not find {key} at {fallback}")


def save_result_directory(
    output_dir: str | Path,
    predictions: Dict[str, torch.Tensor],
    *,
    frame_dir: str | Path,
    image_paths: Sequence[str],
    model_name: str,
    model_kind: str,
    target_resolution: Optional[Tuple[int, int]],
    forward_kwargs: Dict[str, Any],
    source_video: Optional[str] = None,
    inference_stats: Optional[Dict[str, Any]] = None,
    overwrite: bool = True,
) -> Dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    frame_dir = Path(frame_dir).resolve()
    if overwrite and output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred: Dict[str, torch.Tensor] = {}
    for key, value in predictions.items():
        if not torch.is_tensor(value):
            continue
        pred[key] = value.squeeze(0) if value.ndim > 0 and value.shape[0] == 1 else value

    conf = pred["conf"]
    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf.squeeze(-1)
    conf = conf.to(torch.float16).contiguous()

    camera_poses = pred["camera_poses"].to(torch.float32).contiguous()
    local_points = pred.get("local_points")
    if local_points is None:
        raise RuntimeError("Expected local_points in predictions for depth export.")

    if "points" in pred:
        points = pred["points"].to(torch.float16).contiguous()
    else:
        points = torch.einsum(
            "nij, nhwj -> nhwi",
            camera_poses,
            homogenize_points(local_points.to(torch.float32)),
        )[..., :3].to(torch.float16).contiguous()

    depth_maps = local_points[..., 2].to(torch.float16).contiguous()
    target_width, target_height = _resolve_target_resolution(target_resolution, points)

    file_map = {
        "points": output_dir / "points.pt",
        "conf": output_dir / "conf.pt",
        "camera_poses": output_dir / "camera_poses.pt",
        "depth_maps": output_dir / "depth_maps.pt",
    }
    payloads = {
        "points": points,
        "conf": conf,
        "camera_poses": camera_poses,
        "depth_maps": depth_maps,
    }
    for key, path in file_map.items():
        torch.save(payloads[key], path)

    alignment_keys = [
        "chunk_sim3_scales",
        "chunk_sim3_poses",
        "chunk_se3_poses",
        "alignment_mode",
        "metric",
        "overlap_prev_cam",
        "overlap_next_cam",
        "overlap_prev_pcd",
        "overlap_next_pcd",
        "overlap_next_conf",
    ]
    alignment_payload = {key: pred[key] for key in alignment_keys if key in pred}
    if alignment_payload:
        torch.save(alignment_payload, output_dir / "alignment.pt")

    trajectory_plot = _save_trajectory_xz_plot(output_dir / "trajectory_xz.png", camera_poses)
    meta = {
        "frame_dir": str(frame_dir),
        "source_video": source_video,
        "num_frames": int(camera_poses.shape[0]),
        "target_resolution": [target_width, target_height],
        "model_name": model_name,
        "model_kind": model_kind,
        "forward_kwargs": forward_kwargs,
        "image_count": len(image_paths),
        "files": {key: str(path) for key, path in file_map.items()},
        "trajectory_plot": str(trajectory_plot),
        "inference_stats": inference_stats or {},
    }
    with open(output_dir / "meta.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(meta, handle, sort_keys=False)
    return meta


def load_result_meta(result_dir: str | Path) -> Dict[str, Any]:
    result_dir = Path(result_dir).resolve()
    with open(result_dir / "meta.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_result_tensors(result_dir: str | Path) -> Dict[str, torch.Tensor]:
    result_dir = Path(result_dir).resolve()
    meta = load_result_meta(result_dir)
    return {
        "points": torch.load(_resolve_result_file(result_dir, meta, "points", "points.pt"), map_location="cpu", weights_only=False),
        "conf": torch.load(_resolve_result_file(result_dir, meta, "conf", "conf.pt"), map_location="cpu", weights_only=False),
        "camera_poses": torch.load(_resolve_result_file(result_dir, meta, "camera_poses", "camera_poses.pt"), map_location="cpu", weights_only=False),
        "depth_maps": torch.load(_resolve_result_file(result_dir, meta, "depth_maps", "depth_maps.pt"), map_location="cpu", weights_only=False),
    }


def load_result_for_viser(result_dir: str | Path, *, verbose: bool = True) -> Dict[str, Any]:
    result_dir = Path(result_dir).resolve()
    meta = load_result_meta(result_dir)
    frame_dir = Path(meta["frame_dir"])
    target_resolution = tuple(meta["target_resolution"])
    image_paths = list_image_files(frame_dir)
    images = load_images_from_paths(image_paths, target_resolution=target_resolution, verbose=verbose)
    tensors = load_result_tensors(result_dir)
    return {
        "images": images.permute(0, 2, 3, 1).numpy(),
        "points": tensors["points"].float().numpy(),
        "conf": tensors["conf"].float().numpy(),
        "camera_poses": tensors["camera_poses"].float().numpy(),
        "frame_dir": str(frame_dir),
        "target_resolution": list(target_resolution),
        "window_size": int((meta.get("forward_kwargs") or {}).get("window_size", 32)),
        "overlap_size": int((meta.get("forward_kwargs") or {}).get("overlap_size", 3)),
    }


__all__ = [
    "list_image_files",
    "load_images_from_paths",
    "load_result_for_viser",
    "load_result_meta",
    "load_result_tensors",
    "save_result_directory",
]
