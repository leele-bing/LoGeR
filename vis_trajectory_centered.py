from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 2D plane trajectory centered on any chosen frame.")
    parser.add_argument("--input", type=str, required=True, help="Path to camera_poses.npz/.pt or result dir containing them")
    parser.add_argument("--center_frame", type=int, default=0, help="Index of the frame to be used as the coordinate center")
    parser.add_argument("--align_orientation", action="store_true", help="If set, rotate so the chosen frame becomes the identity pose (both translate and rotate)")
    parser.add_argument("--plane", type=str, default="xz", choices=["xz", "xy", "yz"], help="Plane to plot")
    parser.add_argument("--half_window", type=int, default=None, help="Number of frames before/after center to include (e.g. 200)")
    parser.add_argument(
        "--remap_zx_to_xy",
        action="store_true",
        help="When plotting the xz plane, remap coordinates as X'=Z and Y'=-X for display.",
    )
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    parser.add_argument("--dpi", type=int, default=160, help="Figure DPI")
    return parser.parse_args()


def resolve_camera_pose_file(input_path: str | Path) -> Path:
    path = Path(input_path).expanduser().resolve()
    if path.is_file():
        return path
    if path.is_dir():
        for filename in ("camera_poses.npz", "camera_poses.pt"):
            p = path / filename
            if p.is_file():
                return p
    raise FileNotFoundError(f"Cannot find camera_poses.npz or camera_poses.pt from input: {path}")


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
    normalized = torch.stack([normalize_pose_matrix(p) for p in poses], dim=0)
    return normalized[:, :3, 3]


def poses_apply_transform(camera_poses: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    # transform is 4x4; apply to each pose: T @ pose
    poses = camera_poses
    if poses.ndim == 4 and poses.shape[0] == 1:
        poses = poses.squeeze(0)
    if poses.ndim != 3:
        raise ValueError(f"Expected camera_poses with shape [N, 3/4, 4/4], got {tuple(poses.shape)}")
    normalized = torch.stack([normalize_pose_matrix(p) for p in poses], dim=0)
    T = transform.to(normalized.dtype)
    transformed = torch.stack([T @ p for p in normalized], dim=0)
    return transformed


def axis_indices_for_plane(plane: str) -> Tuple[int, int, str, str]:
    if plane == "xy":
        return 0, 1, "X", "Y"
    if plane == "yz":
        return 1, 2, "Y", "Z"
    return 0, 2, "X", "Z"


def remap_xz_to_xy(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str, str]:
    # For display only: map original (X, Z) to (X', Y') = (Z, -X).
    return b, -a, "Z", "-X"


def set_equal_axes_2d(ax: plt.Axes, a: np.ndarray, b: np.ndarray) -> None:
    a_min, a_max = float(np.min(a)), float(np.max(a))
    b_min, b_max = float(np.min(b)), float(np.max(b))
    a_center = 0.5 * (a_min + a_max)
    b_center = 0.5 * (b_min + b_max)
    half_extent = max(a_max - a_min, b_max - b_min, 1e-6) * 0.5
    ax.set_xlim(a_center - half_extent, a_center + half_extent)
    ax.set_ylim(b_center - half_extent, b_center + half_extent)
    ax.set_aspect("equal", adjustable="box")


def save_centered_plane_plot(
    output_path: Path,
    camera_poses: torch.Tensor,
    center_idx: int,
    align_orientation: bool,
    plane: str,
    dpi: int,
    half_window: int | None = None,
    remap_zx_to_xy: bool = False,
) -> Path:
    # Prepare poses: optionally apply inverse of chosen frame so it's identity
    poses = camera_poses
    if poses.ndim == 4 and poses.shape[0] == 1:
        poses = poses.squeeze(0)

    if center_idx < 0 or center_idx >= poses.shape[0]:
        raise IndexError(f"center_frame {center_idx} out of range [0, {poses.shape[0]-1}]")

    normalized = torch.stack([normalize_pose_matrix(p) for p in poses], dim=0)

    if align_orientation:
        center_pose = normalized[center_idx].to(torch.float64)
        T_inv = torch.linalg.inv(center_pose)
        transformed = torch.stack([T_inv @ p.to(torch.float64) for p in normalized], dim=0).to(normalized.dtype)
    else:
        # Only translate so that center becomes origin: subtract its translation
        centers = normalized[:, :3, 3]
        center_trans = centers[center_idx]
        transformed = normalized.clone()
        transformed[:, :3, 3] = transformed[:, :3, 3] - center_trans

    centers_new = camera_centers_from_poses(transformed).detach().cpu().float().numpy()
    if centers_new.size == 0:
        raise RuntimeError("Cannot plot trajectory without camera centers.")

    N = centers_new.shape[0]
    start_i = 0
    end_i = N
    if half_window is not None:
        start_i = max(0, center_idx - half_window)
        end_i = min(N, center_idx + half_window + 1)

    ia, ib, la, lb = axis_indices_for_plane(plane)
    centers_slice = centers_new[start_i:end_i]
    a = centers_slice[:, ia]
    b = centers_slice[:, ib]

    if remap_zx_to_xy and plane == "xz":
        a, b, la, lb = remap_xz_to_xy(a, b)

    colors = plt.get_cmap("viridis")(np.linspace(0.0, 1.0, len(a)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    ax.plot(a, b, color="0.35", linewidth=1.5, alpha=0.8)
    ax.scatter(a, b, c=colors, s=14, linewidths=0)
    # Highlight center, start and end within the slice
    rel_center = center_idx - start_i
    if 0 <= rel_center < len(a):
        ax.scatter([a[rel_center]], [b[rel_center]], c=["#2ca02c"], s=64, label="center", zorder=4)
    ax.scatter([a[0]], [b[0]], c=["#2ca02c"], s=48, label="start", zorder=3)
    ax.scatter([a[-1]], [b[-1]], c=["#d62728"], s=48, label="end", zorder=3)
    ax.set_xlabel(la)
    ax.set_ylabel(lb)
    title = f"Trajectory on {la}{lb} plane (center_frame={center_idx}, align_orientation={align_orientation}, remap_zx_to_xy={remap_zx_to_xy})"
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    set_equal_axes_2d(ax, a, b)
    ax.legend(loc="best")
    ax.text(0.02, 0.02, f"frames: {N} (showing {len(a)} from {start_i} to {end_i-1})", transform=ax.transAxes, fontsize=9, color="0.35", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    pose_path = resolve_camera_pose_file(args.input)
    camera_poses = load_camera_poses(pose_path)

    if args.output is None:
        output_path = pose_path.parent / f"trajectory_centered_{args.center_frame}_{args.plane}.png"
    else:
        output_path = Path(args.output).expanduser().resolve()

    saved = save_centered_plane_plot(
        output_path,
        camera_poses,
        args.center_frame,
        args.align_orientation,
        args.plane,
        dpi=args.dpi,
        half_window=args.half_window,
        remap_zx_to_xy=args.remap_zx_to_xy,
    )
    print(f"Loaded poses: {pose_path}")
    print(f"Saved centered trajectory plot: {saved}")


if __name__ == "__main__":
    main()
