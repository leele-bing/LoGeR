from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot trajectory from camera_poses.pt.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to camera_poses.pt, or a result directory that contains camera_poses.pt.",
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
    return parser.parse_args()


def resolve_camera_pose_file(input_path: str | Path) -> Path:
    path = Path(input_path).expanduser().resolve()
    if path.is_file():
        return path
    if path.is_dir():
        pose_path = path / "camera_poses.pt"
        if pose_path.is_file():
            return pose_path
    raise FileNotFoundError(f"Cannot find camera_poses.pt from input: {path}")


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


def axes_for_plane(plane: str) -> Tuple[int, int, str, str]:
    if plane == "xy":
        return 0, 1, "X", "Y"
    if plane == "yz":
        return 1, 2, "Y", "Z"
    return 0, 2, "X", "Z"


def save_trajectory_plot(output_path: Path, camera_poses: torch.Tensor, plane: str, dpi: int) -> Path:
    centers = camera_centers_from_poses(camera_poses).detach().cpu().float().numpy()
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
    ax.set_title(f"Camera Trajectory on {label_a}{label_b} Plane")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best")
    ax.text(0.02, 0.02, f"frames: {len(centers)}", transform=ax.transAxes, fontsize=9, color="0.35", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_trajectory_plot_3d(output_path: Path, camera_poses: torch.Tensor, dpi: int) -> Path:
    centers = camera_centers_from_poses(camera_poses).detach().cpu().float().numpy()
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
    ax.set_title("Camera Trajectory in 3D Space")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.show()
    return output_path


def main() -> None:
    args = parse_args()
    pose_path = resolve_camera_pose_file(args.input)
    camera_poses = torch.load(pose_path, map_location="cpu", weights_only=False)
    if not torch.is_tensor(camera_poses):
        raise TypeError(f"Expected tensor in {pose_path}, got {type(camera_poses)!r}")

    if args.output is None:
        if args.mode == "3d":
            output_path = pose_path.parent / "trajectory_3d.png"
        else:
            output_path = pose_path.parent / f"trajectory_{args.plane}.png"
    else:
        output_path = Path(args.output).expanduser().resolve()

    if args.mode == "3d":
        saved = save_trajectory_plot_3d(output_path, camera_poses, dpi=args.dpi)
    else:
        saved = save_trajectory_plot(output_path, camera_poses, plane=args.plane, dpi=args.dpi)
    print(f"Loaded poses: {pose_path}")
    print(f"Saved trajectory plot: {saved}")


if __name__ == "__main__":
    main()
