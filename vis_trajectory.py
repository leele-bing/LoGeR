from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_utils import load_result_meta
from align_ground import apply_transform_to_poses, estimate_trajectory_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot trajectory from camera_poses data.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to camera_poses.pt/.npz, or a result directory that contains camera_poses data.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path. Default: trajectory_3d.png for 3D mode, trajectory_<plane>.png for 2D mode.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="3d",
        choices=["2d", "3d"],
        help="Plot mode. 3d draws full XYZ trajectory; 2d draws a projected plane.",
    )
    parser.add_argument(
        "--plane",
        type=str,
        default="xz",
        choices=["xz", "xy", "yz"],
        help="Projection plane for plotting.",
    )
    parser.add_argument("--dpi", type=int, default=160, help="Figure DPI.")
    parser.add_argument("--use_result_frame", action="store_true", help="Use the saved result frame instead of recentering to the first frame.")
    parser.add_argument(
        "--reference_frame",
        type=str,
        default="auto",
        choices=["auto", "initial_camera", "result", "trajectory_plane"],
        help="Which reference frame to plot in.",
    )
    parser.add_argument(
        "--plane_origin",
        type=str,
        default="first_camera",
        choices=["first_camera", "centroid"],
        help="Origin placement used when reference_frame=trajectory_plane in the trajectory-aligned frame.",
    )
    parser.add_argument("--frame_step", type=int, default=1, help="Use the first N frames to estimate up, and the displacement from frame 0 to frame N to define forward in the trajectory-aligned frame.")
    return parser.parse_args()


def resolve_camera_pose_file(input_path: str | Path) -> Path:
    path = Path(input_path).expanduser().resolve()
    if path.is_file():
        return path
    if path.is_dir():
        for filename in ("camera_poses.npz", "camera_poses.pt"):
            pose_path = path / filename
            if pose_path.is_file():
                return pose_path
    raise FileNotFoundError(f"Cannot find camera_poses.npz or camera_poses.pt from input: {path}")


def load_meta_from_input(input_path: str | Path) -> dict:
    path = Path(input_path).expanduser().resolve()
    if path.is_dir() and (path / "meta.yaml").is_file():
        return load_result_meta(path)
    return {}


def load_camera_poses(pose_path: str | Path) -> torch.Tensor:
    path = Path(pose_path).expanduser().resolve()
    if path.suffix == ".npz":
        with np.load(path) as payload:
            if "camera_poses" not in payload:
                raise KeyError(f"Key 'camera_poses' not found in {path}")
            camera_poses = torch.from_numpy(payload["camera_poses"])
    else:
        camera_poses = torch.load(path, map_location="cpu", weights_only=False)
    if not torch.is_tensor(camera_poses):
        raise TypeError(f"Expected tensor in {path}, got {type(camera_poses)!r}")
    return camera_poses


def normalize_pose_matrix(pose: torch.Tensor) -> torch.Tensor:
    if pose.shape == (4, 4):
        return pose
    if pose.shape == (3, 4):
        full = torch.eye(4, dtype=pose.dtype, device=pose.device)
        full[:3, :] = pose
        return full
    raise ValueError(f"Unexpected pose shape: {tuple(pose.shape)}")


def camera_centers_from_poses(camera_poses: torch.Tensor) -> torch.Tensor:
    poses = camera_poses
    if poses.ndim == 4 and poses.shape[0] == 1:
        poses = poses.squeeze(0)
    if poses.ndim != 3:
        raise ValueError(f"Expected camera_poses with shape [N, 3/4, 4/4], got {tuple(poses.shape)}")
    normalized = torch.stack([normalize_pose_matrix(pose) for pose in poses], dim=0)
    return normalized[:, :3, 3]


def poses_in_initial_camera_frame(camera_poses: torch.Tensor) -> torch.Tensor:
    poses = camera_poses
    if poses.ndim == 4 and poses.shape[0] == 1:
        poses = poses.squeeze(0)
    if poses.ndim != 3:
        raise ValueError(f"Expected camera_poses with shape [N, 3/4, 4/4], got {tuple(poses.shape)}")

    normalized = torch.stack([normalize_pose_matrix(pose) for pose in poses], dim=0)
    first_pose_inv = torch.linalg.inv(normalized[0].to(torch.float64)).to(normalized.dtype)
    return torch.stack([first_pose_inv @ pose for pose in normalized], dim=0)


def axes_for_plane(plane: str) -> Tuple[int, int, str, str]:
    if plane == "xy":
        return 0, 1, "X", "Y"
    if plane == "yz":
        return 1, 2, "Y", "Z"
    return 0, 2, "X", "Z"


def set_equal_axes_2d(ax: plt.Axes, a: np.ndarray, b: np.ndarray) -> None:
    a_min, a_max = float(np.min(a)), float(np.max(a))
    b_min, b_max = float(np.min(b)), float(np.max(b))
    a_center = 0.5 * (a_min + a_max)
    b_center = 0.5 * (b_min + b_max)
    half_extent = max(a_max - a_min, b_max - b_min, 1e-6) * 0.5
    ax.set_xlim(a_center - half_extent, a_center + half_extent)
    ax.set_ylim(b_center - half_extent, b_center + half_extent)
    ax.set_aspect("equal", adjustable="box")


def set_equal_axes_3d(ax, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    z_min, z_max = float(np.min(z)), float(np.max(z))

    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    z_center = 0.5 * (z_min + z_max)
    half_extent = max(x_max - x_min, y_max - y_min, z_max - z_min, 1e-6) * 0.5

    ax.set_xlim(x_center - half_extent, x_center + half_extent)
    ax.set_ylim(y_center - half_extent, y_center + half_extent)
    ax.set_zlim(z_center - half_extent, z_center + half_extent)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def save_trajectory_plot(
    output_path: Path,
    camera_poses: torch.Tensor,
    plane: str,
    dpi: int,
    *,
    canonical_first_frame: bool = True,
) -> Path:
    poses = poses_in_initial_camera_frame(camera_poses) if canonical_first_frame else camera_poses
    centers = camera_centers_from_poses(poses).detach().cpu().float().numpy()
    if centers.size == 0:
        raise RuntimeError("Cannot plot trajectory without camera centers.")

    axis_a, axis_b, label_a, label_b = axes_for_plane(plane)
    a = centers[:, axis_a]
    b = centers[:, axis_b]
    colors = plt.get_cmap("viridis")(np.linspace(0.0, 1.0, len(centers)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    ax.plot(a, b, color="0.35", linewidth=1.5, alpha=0.8)
    ax.scatter(a, b, c=colors, s=14, linewidths=0)
    ax.scatter([a[0]], [b[0]], c=["#2ca02c"], s=48, label="start", zorder=3)
    ax.scatter([a[-1]], [b[-1]], c=["#d62728"], s=48, label="end", zorder=3)
    ax.set_xlabel(label_a)
    ax.set_ylabel(label_b)
    title_suffix = "Initial Camera Frame" if canonical_first_frame else "Result Frame"
    ax.set_title(f"Camera Trajectory on {label_a}{label_b} Plane ({title_suffix})")
    ax.grid(True, alpha=0.25)
    set_equal_axes_2d(ax, a, b)
    ax.legend(loc="best")
    ax.text(0.02, 0.02, f"frames: {len(centers)}", transform=ax.transAxes, fontsize=9, color="0.35", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_trajectory_plot_3d(
    output_path: Path,
    camera_poses: torch.Tensor,
    dpi: int,
    *,
    canonical_first_frame: bool = True,
) -> Path:
    poses = poses_in_initial_camera_frame(camera_poses) if canonical_first_frame else camera_poses
    centers = camera_centers_from_poses(poses).detach().cpu().float().numpy()
    if centers.size == 0:
        raise RuntimeError("Cannot plot trajectory without camera centers.")

    x = centers[:, 0]
    y = centers[:, 1]
    z = centers[:, 2]
    colors = plt.get_cmap("viridis")(np.linspace(0.0, 1.0, len(centers)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9, 7), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, color="0.35", linewidth=1.5, alpha=0.8)
    ax.scatter(x, y, z, c=colors, s=10, depthshade=True)
    ax.scatter([x[0]], [y[0]], [z[0]], c=["#2ca02c"], s=64, label="start", depthshade=False)
    ax.scatter([x[-1]], [y[-1]], [z[-1]], c=["#d62728"], s=64, label="end", depthshade=False)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    title_suffix = "Initial Camera Frame" if canonical_first_frame else "Result Frame"
    ax.set_title(f"Camera Trajectory in 3D Space ({title_suffix})")
    set_equal_axes_3d(ax, x, y, z)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.show()
    return output_path


def main() -> None:
    args = parse_args()
    meta = load_meta_from_input(args.input)
    pose_path = resolve_camera_pose_file(args.input)
    camera_poses = load_camera_poses(pose_path)
    if args.reference_frame != "auto":
        reference_frame = args.reference_frame
    elif args.use_result_frame or meta.get("reference_frame", "initial_camera") != "initial_camera":
        reference_frame = "result"
    else:
        reference_frame = "initial_camera"
    if reference_frame == "trajectory_plane":
        alignment = estimate_trajectory_frame(
            camera_poses,
            frame_step=args.frame_step,
            plane_origin=args.plane_origin,
        )
        camera_poses = apply_transform_to_poses(camera_poses, alignment["transform"])
    canonical_first_frame = reference_frame == "initial_camera"

    if args.output is None:
        if args.mode == "3d":
            output_path = pose_path.parent / "trajectory_3d.png"
        else:
            output_path = pose_path.parent / f"trajectory_{args.plane}.png"
    else:
        output_path = Path(args.output).expanduser().resolve()

    if args.mode == "3d":
        saved = save_trajectory_plot_3d(output_path, camera_poses, dpi=args.dpi, canonical_first_frame=canonical_first_frame)
    else:
        saved = save_trajectory_plot(output_path, camera_poses, plane=args.plane, dpi=args.dpi, canonical_first_frame=canonical_first_frame)
    print(f"Loaded poses: {pose_path}")
    print(f"Saved trajectory plot: {saved}")


if __name__ == "__main__":
    main()
