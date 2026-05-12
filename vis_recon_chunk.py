from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib.cm as cm
import numpy as np
from tqdm.auto import tqdm
import viser
import viser.transforms as vt

from align_ground import apply_transform_to_points, apply_transform_to_poses, estimate_trajectory_frame
from data_utils import load_result_for_viser, load_result_meta
from loger.utils.viser_utils import add_origin_axes, add_scene_grid, apply_sky_segmentation, setup_camera_follow


REQUIRED_RESULT_FILES = (
    "meta.yaml",
    "camera_poses.npz",
    "depth_maps.npz",
    "points.pt",
    "conf.npz",
    "trajectory_xz.png",
)


@dataclass(frozen=True)
class ChunkInfo:
    chunk_idx: int
    name: str
    result_dir: Path
    frame_dir: Path
    num_frames: int


@dataclass
class ChunkState:
    info: ChunkInfo
    pred_dict: dict
    cam_ids: List[str]
    root_handle: object
    frames_roots: Dict[str, List[object]]
    pcd_handles: Dict[str, List[object]]
    frustums: Dict[str, List[object]]
    images: Dict[str, np.ndarray]
    points: Dict[str, np.ndarray]
    conf: Dict[str, np.ndarray]
    camera_poses: Dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize reconstruction chunks lazily in one viser app.")
    parser.add_argument("--result_root", type=str, required=True, help="Video-level traj directory, e.g. /data/xby/YTB/shanghai0/traj/BV14p4y1876e")
    parser.add_argument("--frame_root", type=str, default=None, help="Matching video-level img directory. If omitted, infer it from result_root by replacing /traj/ with /img/.")

    parser.add_argument("--port", type=int, default=8080, help="Port number for the viser server.")
    parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode.")
    parser.add_argument("--share", action="store_true", help="Share the viser server with others.")
    parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation per loaded chunk.")

    parser.add_argument("--subsample", type=int, default=5, help="Point cloud subsample factor.")
    parser.add_argument("--conf_threshold", type=float, default=20.0, help="Initial confidence threshold.")
    parser.add_argument("--video_width", type=int, default=320, help="Video preview width in the GUI.")
    parser.add_argument("--start_chunk_idx", type=int, default=0, help="First chunk index to include.")
    parser.add_argument("--end_chunk_idx", type=int, default=None, help="End chunk index (exclusive).")
    parser.add_argument("--max_chunks", type=int, default=None, help="Optional limit on how many chunks to expose.")
    parser.add_argument(
        "--max_cached_chunks",
        type=int,
        default=1,
        help="Maximum number of loaded chunks kept in memory. Use 1 to only keep the active chunk.",
    )
    parser.add_argument(
        "--reference_frame",
        type=str,
        default="auto",
        choices=["auto", "initial_camera", "result", "trajectory_plane"],
        help="Which reference frame to visualize in.",
    )
    parser.add_argument("--frame_step", type=int, default=10, help="Use the first N frames to estimate up, and the displacement from frame 0 to frame N to define forward in the trajectory-aligned frame.")
    return parser.parse_args()


def infer_frame_root(result_root: Path) -> Path:
    parts = list(result_root.resolve().parts)
    if "traj" not in parts:
        raise ValueError("Could not infer frame_root automatically because result_root does not contain a 'traj' path component.")
    traj_idx = parts.index("traj")
    inferred_parts = parts[:]
    inferred_parts[traj_idx] = "img"
    return Path(*inferred_parts)


def is_result_dir(path: Path) -> bool:
    return path.is_dir() and all((path / name).exists() for name in REQUIRED_RESULT_FILES)


def list_chunk_result_dirs(result_root: Path) -> List[Path]:
    result_root = result_root.expanduser().resolve()
    if is_result_dir(result_root):
        return [result_root]
    return sorted(path for path in result_root.iterdir() if is_result_dir(path))


def resolve_frame_dir(chunk_result_dir: Path, frame_root: Path) -> Path:
    candidate = frame_root / chunk_result_dir.name
    if candidate.is_dir():
        return candidate

    parts = list(chunk_result_dir.resolve().parts)
    if "traj" not in parts:
        raise FileNotFoundError(f"Could not infer frame directory for {chunk_result_dir}")
    traj_idx = parts.index("traj")
    inferred_parts = parts[:]
    inferred_parts[traj_idx] = "img"
    inferred_dir = Path(*inferred_parts)
    if inferred_dir.is_dir():
        return inferred_dir
    raise FileNotFoundError(f"Frame directory not found for {chunk_result_dir}")


def _normalize_pose_matrix(pose: np.ndarray) -> np.ndarray:
    if pose.shape == (4, 4):
        return pose
    if pose.shape == (3, 4):
        full = np.eye(4, dtype=pose.dtype)
        full[:3, :] = pose
        return full
    raise ValueError(f"Unexpected pose shape: {pose.shape}")


def _resolve_reference_frame(args: argparse.Namespace, pred_dict: dict) -> str:
    if args.reference_frame != "auto":
        return args.reference_frame
    if pred_dict.get("use_result_frame", False):
        return "result"
    return "initial_camera"


def transform_payload_for_reference_frame(payload: dict, reference_frame: str, frame_step: int) -> dict:
    if reference_frame == "result":
        return payload

    updated = dict(payload)
    if reference_frame == "trajectory_plane":
        alignment = estimate_trajectory_frame(
            updated["camera_poses"],
            frame_step=frame_step,
            plane_origin="first_camera",
        )
        updated["points"] = apply_transform_to_points(updated["points"], alignment["transform"])
        updated["camera_poses"] = apply_transform_to_poses(updated["camera_poses"], alignment["transform"])
        return updated

    camera_poses = np.asarray(updated["camera_poses"])
    if camera_poses.ndim != 3 or camera_poses.shape[0] == 0:
        return updated

    normalized_poses = np.stack([_normalize_pose_matrix(pose) for pose in camera_poses], axis=0)
    first_pose_inv = np.linalg.inv(normalized_poses[0].astype(np.float64)).astype(normalized_poses.dtype)
    recentered_poses = np.stack([first_pose_inv @ pose for pose in normalized_poses], axis=0)

    points = np.asarray(updated["points"])
    original_shape = points.shape
    points_flat = points.reshape(-1, 3)
    valid_mask = np.all(np.isfinite(points_flat), axis=1)
    transformed_flat = points_flat.copy()
    if np.any(valid_mask):
        valid_points = points_flat[valid_mask]
        rotated = (first_pose_inv[:3, :3] @ valid_points.T).T + first_pose_inv[:3, 3]
        transformed_flat[valid_mask] = rotated.astype(points.dtype, copy=False)
    updated["points"] = transformed_flat.reshape(original_shape)
    updated["camera_poses"] = recentered_poses.astype(camera_poses.dtype, copy=False)
    return updated


def build_chunk_infos(result_root: Path, frame_root: Path, args: argparse.Namespace) -> List[ChunkInfo]:
    chunk_result_dirs = list_chunk_result_dirs(result_root)
    if not chunk_result_dirs:
        raise FileNotFoundError(f"No chunk result directories found in {result_root}")

    start_idx = max(0, args.start_chunk_idx)
    end_idx = len(chunk_result_dirs) if args.end_chunk_idx is None else min(len(chunk_result_dirs), args.end_chunk_idx)
    chunk_result_dirs = chunk_result_dirs[start_idx:end_idx]
    if args.max_chunks is not None:
        chunk_result_dirs = chunk_result_dirs[: args.max_chunks]
    if not chunk_result_dirs:
        raise RuntimeError("No chunks selected after applying the requested chunk range.")

    infos: List[ChunkInfo] = []
    for chunk_idx, result_dir in enumerate(chunk_result_dirs):
        frame_dir = resolve_frame_dir(result_dir, frame_root)
        meta = load_result_meta(result_dir)
        infos.append(
            ChunkInfo(
                chunk_idx=chunk_idx,
                name=result_dir.name,
                result_dir=result_dir,
                frame_dir=frame_dir,
                num_frames=int(meta.get("num_frames", 0)),
            )
        )
    return infos


def load_chunk_payload(info: ChunkInfo, args: argparse.Namespace) -> dict:
    payload = load_result_for_viser(info.result_dir, info.frame_dir, start_frame=0, end_frame=-1, verbose=False)
    reference_frame = _resolve_reference_frame(args, payload)
    payload = transform_payload_for_reference_frame(payload, reference_frame, args.frame_step)
    return payload


def _camera_ids_from_payload(payload: dict) -> List[str]:
    cam_ids = ["cam0"]
    for i in range(1, 6):
        cam_id = f"cam{i:02d}"
        if cam_id in payload:
            cam_ids.append(cam_id)
    return cam_ids


def _process_video_frame(images: np.ndarray, frame_idx: int, video_width: int) -> np.ndarray:
    frame = images[frame_idx]
    if frame.max() <= 1.0:
        frame = (frame * 255).astype(np.uint8)
    else:
        frame = frame.astype(np.uint8)
    h, w = frame.shape[:2]
    new_w = video_width
    new_h = int(h * (new_w / w))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def main() -> None:
    args = parse_args()
    result_root = Path(args.result_root).expanduser().resolve()
    frame_root = Path(args.frame_root).expanduser().resolve() if args.frame_root is not None else infer_frame_root(result_root)
    chunk_infos = build_chunk_infos(result_root, frame_root, args)
    max_chunk_frames = max(max(info.num_frames, 1) for info in chunk_infos)

    print(f"result_root={result_root}")
    print(f"frame_root={frame_root}")
    print(f"selected_chunks={len(chunk_infos)}")
    for info in chunk_infos:
        print(f"  [{info.chunk_idx}] {info.name} frames={info.num_frames}")

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    if args.share:
        server.request_share_url()
    server.scene.set_up_direction("-y")
    add_scene_grid(server)
    add_origin_axes(server)

    with server.gui.add_folder("Playback"):
        gui_play = server.gui.add_checkbox("Playing", False)
        gui_frame = server.gui.add_slider("Frame", 0, max_chunk_frames - 1, 1, 0)
        gui_next = server.gui.add_button("Next")
        gui_prev = server.gui.add_button("Prev")
        gui_fps = server.gui.add_slider("FPS", 1, 60, 0.1, 20)
        gui_all = server.gui.add_checkbox("Show all frames", True)
        gui_stride = server.gui.add_slider("Stride", 1, max_chunk_frames, 1, min(10, max(1, max_chunk_frames - 1)))

    with server.gui.add_folder("Chunks"):
        gui_chunk = server.gui.add_slider("Chunk", 0, len(chunk_infos) - 1, 1, 0)
        gui_chunk_next = server.gui.add_button("Next Chunk")
        gui_chunk_prev = server.gui.add_button("Prev Chunk")

    with server.gui.add_folder("Visualization"):
        gui_conf = server.gui.add_slider("Confidence Percent", 0, 100, 0.1, args.conf_threshold)
        gui_point_size = server.gui.add_slider("Point Size", 0.0001, 0.05, 0.0001, 0.001)
        gui_camera_size = server.gui.add_slider("Camera Size", 0.01, 0.3, 0.01, 0.03)
        gui_show_frustums = server.gui.add_checkbox("Show Cameras", True)

    with server.gui.add_folder("Subsample"):
        gui_subsample = server.gui.add_slider("Point Cloud Subsample", 1, 10, 1, args.subsample)
        gui_apply_subsample = server.gui.add_button("Apply Subsample")

    video_previews = {
        "cam0": server.gui.add_image(np.zeros((max(1, int(args.video_width * 9 / 16)), args.video_width, 3), dtype=np.uint8), format="jpeg", label="Camera 0")
    }

    chunk_cache: Dict[int, ChunkState] = {}
    cache_order: List[int] = []
    active_chunk_idx = [0]
    camera_follow_state = {"stop": lambda: None, "resume": lambda: None}
    current_subsample = [args.subsample]

    def gen_mask(conf_array: np.ndarray, percent: float) -> np.ndarray:
        if conf_array.size == 0 or not np.any(np.isfinite(conf_array)):
            return conf_array > -np.inf
        thresh = percent / 100.0
        return (conf_array >= thresh) & (conf_array > 1e-5)

    def apply_sky_mask_if_requested(payload: dict, info: ChunkInfo) -> dict:
        if not args.mask_sky:
            return payload
        updated = dict(payload)
        conf = updated["conf"]
        masks = apply_sky_segmentation(conf, str(info.frame_dir), is_conf_scores=True)
        updated["conf"] = conf * masks
        return updated

    def build_chunk_state(info: ChunkInfo) -> ChunkState:
        start_time = time.time()
        print(f"Loading chunk [{info.chunk_idx}] {info.name} ...")
        payload = load_chunk_payload(info, args)
        payload = apply_sky_mask_if_requested(payload, info)
        cam_ids = _camera_ids_from_payload(payload)

        images: Dict[str, np.ndarray] = {}
        points: Dict[str, np.ndarray] = {}
        conf: Dict[str, np.ndarray] = {}
        camera_poses: Dict[str, np.ndarray] = {}

        for cam_id in cam_ids:
            if cam_id == "cam0":
                cam_payload = payload
            else:
                cam_payload = payload[cam_id]

            cam_images = cam_payload["images"]
            if cam_images.shape[-1] != 3:
                cam_images = cam_images.transpose(0, 2, 3, 1)
            images[cam_id] = cam_images
            points[cam_id] = cam_payload["points"]
            conf[cam_id] = cam_payload["conf"]

            poses = cam_payload["camera_poses"]
            if poses.shape[-2:] == (4, 4):
                poses = poses[:, :3, :]
            camera_poses[cam_id] = poses

        chunk_len = min(points[cam_id].shape[0] for cam_id in cam_ids)
        for cam_id in cam_ids:
            images[cam_id] = images[cam_id][:chunk_len]
            points[cam_id] = points[cam_id][:chunk_len]
            conf[cam_id] = conf[cam_id][:chunk_len]
            camera_poses[cam_id] = camera_poses[cam_id][:chunk_len]

        root_handle = server.scene.add_frame(f"/chunks/chunk_{info.chunk_idx}", show_axes=False)
        frames_roots: Dict[str, List[object]] = {cam_id: [] for cam_id in cam_ids}
        pcd_handles: Dict[str, List[object]] = {cam_id: [] for cam_id in cam_ids}
        frustums: Dict[str, List[object]] = {cam_id: [] for cam_id in cam_ids}

        print(f"Building chunk [{info.chunk_idx}] scene nodes ...")
        for frame_idx in tqdm(range(chunk_len), leave=False):
            frame_root = server.scene.add_frame(f"/chunks/chunk_{info.chunk_idx}/frames/t{frame_idx}", show_axes=False)
            for cam_id in cam_ids:
                frames_roots[cam_id].append(frame_root)

                current_img = images[cam_id][frame_idx, :: current_subsample[0], :: current_subsample[0]]
                current_xyz = points[cam_id][frame_idx, :: current_subsample[0], :: current_subsample[0]]
                current_conf = conf[cam_id][frame_idx, :: current_subsample[0], :: current_subsample[0]]
                mask = gen_mask(current_conf, gui_conf.value)
                pts_flat = current_xyz.reshape(-1, 3)
                mask_flat = mask.reshape(-1)
                rgb_img_for_pts = current_img
                if rgb_img_for_pts.max() <= 1.0:
                    rgb_img_for_pts = rgb_img_for_pts * 255
                rgb_flat = rgb_img_for_pts.astype(np.uint8).reshape(-1, 3)
                pts = pts_flat[mask_flat]
                rgb = rgb_flat[mask_flat]

                pcd_handle = server.scene.add_point_cloud(
                    f"/chunks/chunk_{info.chunk_idx}/frames/t{frame_idx}/pc_{cam_id}",
                    pts,
                    rgb,
                    point_size=gui_point_size.value * (current_subsample[0] ** 0.5),
                    point_shape="rounded",
                )
                pcd_handles[cam_id].append(pcd_handle)

                pose_3x4 = camera_poses[cam_id][frame_idx]
                pose_4x4 = np.eye(4)
                pose_4x4[:3, :] = pose_3x4
                T_cam = vt.SE3.from_matrix(pose_4x4)
                h_img_cam, w_img_cam = images[cam_id].shape[-3:-1]
                frustum_handle = server.scene.add_camera_frustum(
                    name=f"/chunks/chunk_{info.chunk_idx}/frames/t{frame_idx}/frustum_{cam_id}",
                    fov=1.047,
                    aspect=w_img_cam / h_img_cam,
                    scale=gui_camera_size.value,
                    wxyz=T_cam.rotation().wxyz,
                    position=T_cam.translation(),
                    color=cm.get_cmap("gist_rainbow")(frame_idx / max(chunk_len - 1, 1))[:3],
                    line_width=2.0,
                )
                frustums[cam_id].append(frustum_handle)

        root_handle.visible = False
        elapsed = round(time.time() - start_time, 2)
        print(f"Chunk [{info.chunk_idx}] {info.name} loaded in {elapsed}s")
        return ChunkState(
            info=info,
            pred_dict=payload,
            cam_ids=cam_ids,
            root_handle=root_handle,
            frames_roots=frames_roots,
            pcd_handles=pcd_handles,
            frustums=frustums,
            images=images,
            points=points,
            conf=conf,
            camera_poses=camera_poses,
        )

    def touch_cache(chunk_idx: int) -> None:
        if chunk_idx in cache_order:
            cache_order.remove(chunk_idx)
        cache_order.append(chunk_idx)

    def safe_remove_handle(handle: object) -> None:
        remove_fn = getattr(handle, "remove", None)
        if callable(remove_fn):
            try:
                remove_fn()
                return
            except Exception:
                pass
        if hasattr(handle, "visible"):
            try:
                handle.visible = False
            except Exception:
                pass

    def evict_chunk(chunk_idx: int) -> None:
        if chunk_idx == active_chunk_idx[0]:
            return
        state = chunk_cache.pop(chunk_idx, None)
        if state is None:
            return
        if chunk_idx in cache_order:
            cache_order.remove(chunk_idx)
        safe_remove_handle(state.root_handle)
        state.frames_roots.clear()
        state.pcd_handles.clear()
        state.frustums.clear()
        state.images.clear()
        state.points.clear()
        state.conf.clear()
        state.camera_poses.clear()
        state.cam_ids.clear()
        state.pred_dict.clear()
        print(f"Evicted chunk [{chunk_idx}] from cache")

    def evict_old_chunks_if_needed() -> None:
        if args.max_cached_chunks <= 0:
            return
        while len(chunk_cache) > args.max_cached_chunks:
            evict_candidate = next((idx for idx in cache_order if idx != active_chunk_idx[0]), None)
            if evict_candidate is None:
                break
            evict_chunk(evict_candidate)

    def ensure_chunk_loaded(chunk_idx: int) -> ChunkState:
        if chunk_idx not in chunk_cache:
            chunk_cache[chunk_idx] = build_chunk_state(chunk_infos[chunk_idx])
            cam0_images = chunk_cache[chunk_idx].images["cam0"]
            if "cam0" in video_previews:
                video_previews["cam0"].image = _process_video_frame(cam0_images, 0, args.video_width)
        touch_cache(chunk_idx)
        return chunk_cache[chunk_idx]

    def update_camera_follow_for_chunk(state: ChunkState) -> None:
        camera_follow_state["stop"]()
        cam0_poses = state.camera_poses["cam0"]
        cam0_positions = cam0_poses[:, :3, 3]
        cam0_wxyz = np.array([vt.SO3.from_matrix(pose[:, :3]).wxyz for pose in cam0_poses], dtype=np.float32)
        cam0_forward = np.array([pose[:, :3] @ np.array([0.0, 0.0, 1.0]) for pose in cam0_poses], dtype=np.float32)
        cam0_lookat = cam0_positions + cam0_forward
        stop_follow, resume_follow = setup_camera_follow(
            server=server,
            slider=gui_frame,
            target_positions=cam0_lookat,
            camera_positions=cam0_positions,
            camera_wxyz=cam0_wxyz,
            camera_forward=cam0_forward,
            camera_ema_alpha=lambda: 0.05,
            frame_lag=lambda: 0,
            backoff_distance=lambda: 0.25,
            up_direction=(0.0, -1.0, 0.0),
            fov=60.0,
            frame_offset=0,
        )
        camera_follow_state["stop"] = stop_follow
        camera_follow_state["resume"] = resume_follow

    def refresh_chunk_pointclouds(state: ChunkState) -> None:
        point_size = gui_point_size.value * (current_subsample[0] ** 0.5)
        for cam_id in state.cam_ids:
            for frame_idx in range(len(state.pcd_handles[cam_id])):
                current_img = state.images[cam_id][frame_idx, :: current_subsample[0], :: current_subsample[0]]
                current_xyz = state.points[cam_id][frame_idx, :: current_subsample[0], :: current_subsample[0]]
                current_conf = state.conf[cam_id][frame_idx, :: current_subsample[0], :: current_subsample[0]]
                mask = gen_mask(current_conf, gui_conf.value)
                pts_flat = current_xyz.reshape(-1, 3)
                mask_flat = mask.reshape(-1)
                rgb_img_for_pts = current_img
                if rgb_img_for_pts.max() <= 1.0:
                    rgb_img_for_pts = rgb_img_for_pts * 255
                rgb_flat = rgb_img_for_pts.astype(np.uint8).reshape(-1, 3)
                state.pcd_handles[cam_id][frame_idx].points = pts_flat[mask_flat]
                state.pcd_handles[cam_id][frame_idx].colors = rgb_flat[mask_flat]
                state.pcd_handles[cam_id][frame_idx].point_size = point_size

    def set_active_chunk(chunk_idx: int) -> None:
        active_chunk_idx[0] = chunk_idx
        state = ensure_chunk_loaded(chunk_idx)
        for loaded_idx, loaded_state in chunk_cache.items():
            loaded_state.root_handle.visible = loaded_idx == chunk_idx
        frame_limit = max(0, len(state.images["cam0"]) - 1)
        if gui_frame.value > frame_limit:
            gui_frame.value = frame_limit
        update_camera_follow_for_chunk(state)
        update_visibility()
        update_video_preview()
        evict_old_chunks_if_needed()
        print(f"Activated chunk [{chunk_idx}] {state.info.name}")

    def update_video_preview() -> None:
        state = ensure_chunk_loaded(active_chunk_idx[0])
        frame_limit = len(state.images["cam0"]) - 1
        frame_idx = int(np.clip(gui_frame.value, 0, max(frame_limit, 0)))
        video_previews["cam0"].image = _process_video_frame(state.images["cam0"], frame_idx, args.video_width)

    def update_visibility() -> None:
        state = ensure_chunk_loaded(active_chunk_idx[0])
        chunk_len = len(state.images["cam0"])
        current_frame = int(np.clip(gui_frame.value, 0, max(chunk_len - 1, 0)))
        stride = max(1, int(gui_stride.value))
        for cam_id in state.cam_ids:
            for frame_idx in range(chunk_len):
                visible = ((frame_idx % stride) == 0) if gui_all.value else frame_idx == current_frame
                state.frames_roots[cam_id][frame_idx].visible = visible
                state.pcd_handles[cam_id][frame_idx].visible = visible
                state.frustums[cam_id][frame_idx].visible = visible and gui_show_frustums.value

    ensure_chunk_loaded(0)
    set_active_chunk(0)

    @gui_chunk.on_update
    def _(_):
        set_active_chunk(gui_chunk.value)

    @gui_chunk_next.on_click
    def _(_):
        next_chunk = gui_chunk.value + 1
        if next_chunk > len(chunk_infos) - 1:
            next_chunk = 0
        gui_chunk.value = next_chunk

    @gui_chunk_prev.on_click
    def _(_):
        prev_chunk = gui_chunk.value - 1
        if prev_chunk < 0:
            prev_chunk = len(chunk_infos) - 1
        gui_chunk.value = prev_chunk

    @gui_frame.on_update
    def _(_):
        update_visibility()
        update_video_preview()

    @gui_all.on_update
    def _(_):
        gui_stride.disabled = not gui_all.value
        update_visibility()

    @gui_stride.on_update
    def _(_):
        update_visibility()

    @gui_conf.on_update
    def _(_):
        refresh_chunk_pointclouds(ensure_chunk_loaded(active_chunk_idx[0]))
        update_visibility()

    @gui_point_size.on_update
    def _(_):
        refresh_chunk_pointclouds(ensure_chunk_loaded(active_chunk_idx[0]))
        update_visibility()

    @gui_apply_subsample.on_click
    def _(_):
        new_subsample = int(gui_subsample.value)
        if new_subsample == current_subsample[0]:
            return
        current_subsample[0] = new_subsample
        refresh_chunk_pointclouds(ensure_chunk_loaded(active_chunk_idx[0]))
        update_video_preview()
        update_visibility()
        print(f"Updated point cloud subsample to {new_subsample}")

    @gui_camera_size.on_update
    def _(_):
        state = ensure_chunk_loaded(active_chunk_idx[0])
        for cam_id in state.cam_ids:
            for handle in state.frustums[cam_id]:
                handle.scale = gui_camera_size.value

    @gui_show_frustums.on_update
    def _(_):
        update_visibility()

    @gui_next.on_click
    def _(_):
        state = ensure_chunk_loaded(active_chunk_idx[0])
        gui_frame.value = 0 if gui_frame.value >= len(state.images["cam0"]) - 1 else gui_frame.value + 1

    @gui_prev.on_click
    def _(_):
        state = ensure_chunk_loaded(active_chunk_idx[0])
        gui_frame.value = len(state.images["cam0"]) - 1 if gui_frame.value <= 0 else gui_frame.value - 1

    def loop() -> None:
        prev_time = time.time()
        while True:
            if gui_play.value:
                state = ensure_chunk_loaded(active_chunk_idx[0])
                now = time.time()
                if now - prev_time >= 1.0 / gui_fps.value:
                    next_frame = gui_frame.value + 1
                    if next_frame >= len(state.images["cam0"]):
                        next_frame = 0
                    gui_frame.value = next_frame
                    prev_time = now
            time.sleep(0.005)

    if args.background_mode:
        import threading

        threading.Thread(target=loop, daemon=True).start()
        print(f"Viser server running in background on port {args.port}")
    else:
        print(f"Viser server running in foreground on port {args.port}. Press Ctrl+C to stop.")
        loop()


if __name__ == "__main__":
    main()
