from __future__ import annotations

import glob
import math
import shutil
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


def _poses_in_initial_camera_frame(camera_poses: torch.Tensor) -> torch.Tensor:
    poses = camera_poses
    if poses.ndim == 4 and poses.shape[0] == 1:
        poses = poses.squeeze(0)
    if poses.ndim != 3:
        raise ValueError(f"Expected camera_poses with shape [N, 3/4, 4/4], got {tuple(poses.shape)}")
    normalized = torch.stack([_normalize_pose_matrix(pose) for pose in poses], dim=0)
    first_pose_inv = torch.linalg.inv(normalized[0].to(torch.float64)).to(normalized.dtype)
    return torch.stack([first_pose_inv @ pose for pose in normalized], dim=0)


def _set_equal_axes_2d(ax: plt.Axes, a: np.ndarray, b: np.ndarray) -> None:
    a_min, a_max = float(np.min(a)), float(np.max(a))
    b_min, b_max = float(np.min(b)), float(np.max(b))
    a_center = 0.5 * (a_min + a_max)
    b_center = 0.5 * (b_min + b_max)
    half_extent = max(a_max - a_min, b_max - b_min, 1e-6) * 0.5
    ax.set_xlim(a_center - half_extent, a_center + half_extent)
    ax.set_ylim(b_center - half_extent, b_center + half_extent)
    ax.set_aspect("equal", adjustable="box")


def save_trajectory_xz_plot(
    output_path: str | Path,
    camera_poses: torch.Tensor,
    *,
    canonical_first_frame: bool = True,
    title_suffix: str | None = None,
) -> Path:
    poses = _poses_in_initial_camera_frame(camera_poses) if canonical_first_frame else camera_poses
    centers = _camera_centers_from_poses(poses).detach().cpu().float().numpy()
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
    if title_suffix is None:
        title_suffix = "Initial Camera Frame" if canonical_first_frame else "Result Frame"
    ax.set_title(f"Camera Trajectory on XZ Plane ({title_suffix})")
    ax.grid(True, alpha=0.25)
    _set_equal_axes_2d(ax, x, z)
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
        if not candidate_path.is_absolute():
            candidate_path = result_dir / candidate_path
        if candidate_path.exists():
            return candidate_path
    fallback = result_dir / filename
    if fallback.exists():
        return fallback
    if candidate:
        raise FileNotFoundError(f"Could not find {key} at {candidate} or {fallback}")
    raise FileNotFoundError(f"Could not find {key} at {fallback}")


def _save_npz_tensor(path: Path, key: str, tensor: torch.Tensor) -> None:
    np.savez_compressed(path, **{key: tensor.detach().cpu().numpy()})


def _load_npz_tensor(path: Path, key: str) -> torch.Tensor:
    with np.load(path) as payload:
        if key not in payload:
            raise KeyError(f"Key {key!r} not found in {path}")
        return torch.from_numpy(payload[key])


def _normalize_stride_value(stride: Any) -> int:
    try:
        return max(1, int(stride))
    except (TypeError, ValueError):
        return 1


def save_result_directory(
    output_dir: str | Path,
    predictions: Dict[str, torch.Tensor],
    *,
    frame_dir: str | Path | None,
    image_paths: Optional[Sequence[str]],
    model_name: str,
    model_kind: str,
    target_resolution: Optional[Tuple[int, int]],
    forward_kwargs: Dict[str, Any],
    stride: int = 1,
    input_frame_stride: int = 1,
    conf_threshold: float = 0.0,
    save_alignment: bool = False,
    inference_stats: Optional[Dict[str, Any]] = None,
    canonical_first_frame_for_plot: bool = True,
    extra_meta: Optional[Dict[str, Any]] = None,
    overwrite: bool = True,
) -> Dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred: Dict[str, torch.Tensor] = {}
    for key, value in predictions.items():
        if not torch.is_tensor(value):
            continue
        pred[key] = value.squeeze(0) if value.ndim > 0 and value.shape[0] == 1 else value

    stride = _normalize_stride_value(stride)
    input_frame_stride = _normalize_stride_value(input_frame_stride)
    absolute_stride = input_frame_stride * stride
    source_num_frames = int(pred["camera_poses"].shape[0])
    frame_indices = torch.arange(0, source_num_frames, stride, dtype=torch.long)

    conf = pred["conf"][frame_indices]
    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf.squeeze(-1)
    conf = conf.to(torch.float32).contiguous()
    conf_threshold = float(conf_threshold)
    conf_threshold_normalized = conf_threshold / 100.0 if conf_threshold > 1.0 else conf_threshold
    conf_threshold_normalized = float(np.clip(conf_threshold_normalized, 0.0, 1.0))
    conf_mask = conf >= conf_threshold_normalized
    conf_uint8 = torch.clamp(torch.round(conf * 255.0), 0.0, 255.0).to(torch.uint8).contiguous()

    camera_poses = pred["camera_poses"][frame_indices].to(torch.float32).contiguous()
    local_points = pred.get("local_points")
    if local_points is not None:
        local_points = local_points[frame_indices]

    if "points" in pred:
        points = pred["points"][frame_indices].to(torch.float16).contiguous()
    elif local_points is not None:
        points = torch.einsum(
            "nij, nhwj -> nhwi",
            camera_poses,
            homogenize_points(local_points.to(torch.float32)),
        )[..., :3].to(torch.float16).contiguous()
    else:
        raise RuntimeError("Expected either points or local_points in predictions for point export.")

    if "depth_maps" in pred:
        depth_maps = pred["depth_maps"][frame_indices].to(torch.float16).contiguous()
    elif local_points is not None:
        depth_maps = local_points[..., 2].to(torch.float16).contiguous()
    else:
        raise RuntimeError("Expected either depth_maps or local_points in predictions for depth export.")
    invalid_mask = ~conf_mask
    points[invalid_mask.unsqueeze(-1).expand_as(points)] = torch.nan
    depth_maps[invalid_mask] = torch.nan
    target_width, target_height = _resolve_target_resolution(target_resolution, points)

    file_map = {
        "points": output_dir / "points.pt",
        "conf": output_dir / "conf.npz",
        "camera_poses": output_dir / "camera_poses.npz",
        "depth_maps": output_dir / "depth_maps.npz",
    }
    torch.save(points, file_map["points"])
    _save_npz_tensor(file_map["conf"], "conf", conf_uint8)
    _save_npz_tensor(file_map["camera_poses"], "camera_poses", camera_poses)
    _save_npz_tensor(file_map["depth_maps"], "depth_maps", depth_maps)

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
    if save_alignment and alignment_payload:
        for key, value in list(alignment_payload.items()):
            if not torch.is_tensor(value):
                continue
            if value.ndim > 0 and value.shape[0] == source_num_frames:
                alignment_payload[key] = value[frame_indices]
            elif value.ndim > 1 and value.shape[1] == source_num_frames:
                alignment_payload[key] = value[:, frame_indices]
        torch.save(alignment_payload, output_dir / "alignment.pt")

    save_trajectory_xz_plot(
        output_dir / "trajectory_xz.png",
        camera_poses,
        canonical_first_frame=canonical_first_frame_for_plot,
    )
    video_name = Path(frame_dir).name if frame_dir is not None else None
    meta = {
        "num_frames": int(camera_poses.shape[0]),
        "stride": absolute_stride,
        "video_name": video_name,
        "reference_frame": "initial_camera" if canonical_first_frame_for_plot else "result",
        "conf_threshold": conf_threshold_normalized,
        "conf_storage": "npz_uint8_255",
        "camera_pose_storage": "npz_float32",
        "depth_storage": "npz_float16",
        "save_alignment": bool(save_alignment),
        "target_resolution": [target_width, target_height],
        "model_name": model_name,
        "model_kind": model_kind,
        "window_size": int(forward_kwargs.get("window_size", 32)),
        "overlap_size": int(forward_kwargs.get("overlap_size", 3)),
        "files": {key: path.name for key, path in file_map.items()},
    }
    if inference_stats:
        meta["inference_stats"] = inference_stats
    if extra_meta:
        meta.update(extra_meta)
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
    conf_path = _resolve_result_file(result_dir, meta, "conf", "conf.npz")
    if conf_path.suffix == ".npz":
        conf = _load_npz_tensor(conf_path, "conf")
    else:
        conf = torch.load(conf_path, map_location="cpu", weights_only=False)
    if torch.is_tensor(conf) and conf.dtype == torch.uint8:
        conf = conf.to(torch.float32) / 255.0
    depth_path = _resolve_result_file(result_dir, meta, "depth_maps", "depth_maps.npz")
    if depth_path.suffix == ".npz":
        depth_maps = _load_npz_tensor(depth_path, "depth_maps")
    else:
        depth_maps = torch.load(depth_path, map_location="cpu", weights_only=False)
    camera_pose_path = _resolve_result_file(result_dir, meta, "camera_poses", "camera_poses.npz")
    if camera_pose_path.suffix == ".npz":
        camera_poses = _load_npz_tensor(camera_pose_path, "camera_poses")
    else:
        camera_poses = torch.load(camera_pose_path, map_location="cpu", weights_only=False)
    return {
        "points": torch.load(_resolve_result_file(result_dir, meta, "points", "points.pt"), map_location="cpu", weights_only=False),
        "conf": conf,
        "camera_poses": camera_poses,
        "depth_maps": depth_maps,
    }


def load_alignment_payload(result_dir: str | Path) -> Dict[str, Any]:
    result_dir = Path(result_dir).resolve()
    alignment_path = result_dir / "alignment.pt"
    if not alignment_path.is_file():
        return {}
    payload = torch.load(alignment_path, map_location="cpu", weights_only=False)
    return payload if isinstance(payload, dict) else {}


def load_result_for_viser(
    result_dir: str | Path,
    frame_dir: str | Path,
    *,
    start_frame: int = 0,
    end_frame: int = -1,
    verbose: bool = True,
) -> Dict[str, Any]:
    result_dir = Path(result_dir).resolve()
    meta = load_result_meta(result_dir)
    frame_dir = Path(frame_dir).expanduser().resolve()
    target_resolution = tuple(meta["target_resolution"])
    image_paths = list_image_files(frame_dir)
    tensors = load_result_tensors(result_dir)
    stored_stride = _normalize_stride_value(meta.get("stride", 1))
    indexed_image_paths = image_paths[::stored_stride]

    sequence_length = min(
        len(indexed_image_paths),
        int(tensors["points"].shape[0]),
        int(tensors["conf"].shape[0]),
        int(tensors["camera_poses"].shape[0]),
    )
    if sequence_length <= 0:
        raise RuntimeError("No frames available in the selected result directory.")

    if start_frame < 0:
        raise ValueError(f"start_frame must be >= 0, got {start_frame}")
    start_idx = min(start_frame, sequence_length - 1)
    end_idx = sequence_length if end_frame == -1 else min(max(start_idx + 1, end_frame), sequence_length)
    selected_image_paths = indexed_image_paths[start_idx:end_idx]
    if not selected_image_paths:
        raise RuntimeError(f"No frames selected for range start_frame={start_frame}, end_frame={end_frame}")

    images = load_images_from_paths(selected_image_paths, target_resolution=target_resolution, verbose=verbose)
    return {
        "images": images.permute(0, 2, 3, 1).numpy(),
        "points": tensors["points"][start_idx:end_idx].float().numpy(),
        "conf": tensors["conf"][start_idx:end_idx].float().numpy(),
        "camera_poses": tensors["camera_poses"][start_idx:end_idx].float().numpy(),
        "meta": meta,
        "use_result_frame": meta.get("reference_frame", "initial_camera") != "initial_camera",
        "frame_dir": str(frame_dir),
        "image_paths": [str(path) for path in selected_image_paths],
        "start_frame": start_idx,
        "end_frame": end_idx,
        "target_resolution": list(target_resolution),
        "window_size": int(meta.get("window_size", (meta.get("forward_kwargs") or {}).get("window_size", 32))),
        "overlap_size": int(meta.get("overlap_size", (meta.get("forward_kwargs") or {}).get("overlap_size", 3))),
    }


__all__ = [
    "list_image_files",
    "load_images_from_paths",
    "load_alignment_payload",
    "load_result_for_viser",
    "load_result_meta",
    "load_result_tensors",
    "save_trajectory_xz_plot",
    "save_result_directory",
]
