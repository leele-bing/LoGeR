from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from loger.utils.geometry import homogenize_points, robust_scale_estimation


Window = Tuple[int, int]
PredictionDict = Dict[str, torch.Tensor | str | None]


def compute_windows(num_frames: int, window_size: int, overlap_size: int) -> Tuple[List[Window], int, int]:
    if window_size <= 0 or window_size >= num_frames:
        return [(0, num_frames)], num_frames, 0

    windows: List[Window] = []
    step = max(window_size - overlap_size, 1)
    for start_idx in range(0, num_frames, step):
        end_idx = min(start_idx + window_size, num_frames)
        if end_idx - start_idx >= overlap_size or (end_idx == num_frames and start_idx < num_frames):
            windows.append((start_idx, end_idx))
        if end_idx == num_frames:
            break
    return windows, window_size, overlap_size


def merge_windowed_predictions(
    all_predictions: Sequence[PredictionDict],
    window_size: int,
    overlap_size: int,
) -> PredictionDict:
    if not all_predictions:
        return {}
    if len(all_predictions) == 1:
        return dict(all_predictions[0])

    merged_predictions: PredictionDict = {}
    keys = list(all_predictions[0].keys())
    sequence_keys = {
        "points",
        "local_points",
        "conf",
        "camera_poses",
        "local_camera_poses",
        "camera_qvec",
        "local_camera_qvec",
    }

    for key in keys:
        window_tensors = [pred.get(key, None) for pred in all_predictions]
        if all(t is None for t in window_tensors):
            continue

        if key in sequence_keys:
            result_parts = []

            first = window_tensors[0]
            if first is not None:
                if overlap_size > 0 and first.shape[1] > overlap_size:
                    result_parts.append(first[:, :-overlap_size])
                elif overlap_size <= 0:
                    result_parts.append(first)

            for tensor in window_tensors[1:-1]:
                if tensor is None:
                    continue
                if overlap_size > 0 and tensor.shape[1] > overlap_size:
                    result_parts.append(tensor[:, :-overlap_size])
                elif overlap_size <= 0:
                    result_parts.append(tensor)

            last_tensor = window_tensors[-1]
            if last_tensor is not None:
                result_parts.append(last_tensor)

            if result_parts:
                merged_predictions[key] = torch.cat(result_parts, dim=1)
            else:
                for tensor in reversed(window_tensors):
                    if tensor is not None:
                        merged_predictions[key] = tensor
                        break
        else:
            for tensor in reversed(window_tensors):
                if tensor is not None:
                    merged_predictions[key] = tensor
                    break

    if overlap_size > 0 and len(all_predictions) > 1:
        prev_cam_chunks = []
        next_cam_chunks = []
        prev_pcd_chunks = []
        next_pcd_chunks = []
        next_conf_chunks = []

        for i in range(len(all_predictions) - 1):
            pred_a = all_predictions[i]
            pred_b = all_predictions[i + 1]

            cam_a = pred_a.get("camera_poses", None)
            cam_b = pred_b.get("camera_poses", None)
            lpts_a = pred_a.get("local_points", None)
            lpts_b = pred_b.get("local_points", None)
            conf_b = pred_b.get("conf", None)

            if cam_a is not None and cam_b is not None and cam_a.shape[1] >= overlap_size and cam_b.shape[1] >= overlap_size:
                prev_cam_chunks.append(cam_a[:, cam_a.shape[1] - overlap_size : cam_a.shape[1]])
                next_cam_chunks.append(cam_b[:, 0:overlap_size])

            if lpts_a is not None and lpts_b is not None and lpts_a.shape[1] >= overlap_size and lpts_b.shape[1] >= overlap_size:
                prev_pcd_chunks.append(lpts_a[:, lpts_a.shape[1] - overlap_size : lpts_a.shape[1]])
                next_pcd_chunks.append(lpts_b[:, 0:overlap_size])
                if conf_b is not None and conf_b.shape[1] >= overlap_size:
                    next_conf_chunks.append(conf_b[:, 0:overlap_size].squeeze(-1))

        if prev_cam_chunks and next_cam_chunks:
            merged_predictions["overlap_prev_cam"] = torch.stack(prev_cam_chunks, dim=1)
            merged_predictions["overlap_next_cam"] = torch.stack(next_cam_chunks, dim=1)
        if prev_pcd_chunks and next_pcd_chunks:
            merged_predictions["overlap_prev_pcd"] = torch.stack(prev_pcd_chunks, dim=1)
            merged_predictions["overlap_next_pcd"] = torch.stack(next_pcd_chunks, dim=1)
            if next_conf_chunks:
                merged_predictions["overlap_next_conf"] = torch.stack(next_conf_chunks, dim=1)

    return merged_predictions


def merge_windowed_predictions_sim3(
    all_predictions: Sequence[PredictionDict],
    window_size: int,
    overlap_size: int,
    *,
    allow_scale: bool = True,
    scale_mode: str = "median",
    reset_every: int = 0,
    reuse_transform_within_reset_block: bool = False,
) -> PredictionDict:
    if not all_predictions:
        return {}
    if len(all_predictions) == 1:
        merged = dict(all_predictions[0])
        merged["alignment_mode"] = "sim3" if allow_scale else "se3"
        return merged

    sample_tensor = None
    for pred in all_predictions:
        for key in ("points", "camera_poses", "local_points", "conf"):
            tensor = pred.get(key, None)
            if tensor is not None:
                sample_tensor = tensor
                break
        if sample_tensor is not None:
            break
    if sample_tensor is None:
        raise ValueError("Sim3/SE3 merge requires at least one tensor prediction")

    device = sample_tensor.device
    dtype = sample_tensor.dtype
    batch_size = sample_tensor.shape[0]

    identity_rot = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    zero_trans = torch.zeros(batch_size, 3, device=device, dtype=dtype)
    one_scale = torch.ones(batch_size, device=device, dtype=dtype)

    aligned_predictions: List[PredictionDict] = []
    sim3_scales: Optional[List[torch.Tensor]] = [] if allow_scale else None
    sim3_poses: List[torch.Tensor] = []

    def estimate_relative_transform(prev_aligned: PredictionDict, curr_raw: PredictionDict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if overlap_size <= 0:
            return torch.ones_like(one_scale), identity_rot, zero_trans

        prev_cam = prev_aligned.get("camera_poses", None)
        curr_cam = curr_raw.get("camera_poses", None)
        if prev_cam is None or curr_cam is None or prev_cam.shape[1] == 0 or curr_cam.shape[1] == 0:
            return torch.ones_like(one_scale), identity_rot, zero_trans

        prev_frames = prev_cam.shape[1]
        prev_idx = max(prev_frames - overlap_size, 0)

        prev_pose = prev_cam[:, prev_idx]
        curr_pose = curr_cam[:, 0]

        r_prev = prev_pose[:, :3, :3]
        t_prev = prev_pose[:, :3, 3]
        r_curr = curr_pose[:, :3, :3]
        t_curr = curr_pose[:, :3, 3]

        relative_rot = torch.matmul(r_prev, r_curr.transpose(-1, -2))
        relative_scale = torch.ones_like(one_scale)

        if allow_scale:
            prev_local_raw = prev_aligned.get("local_points", None)
            if prev_local_raw is None:
                prev_local_raw = prev_aligned.get("_local_points_raw", None)
            curr_local_raw = curr_raw.get("local_points", None)

            if (
                prev_local_raw is not None
                and curr_local_raw is not None
                and prev_local_raw.shape[1] > prev_idx
                and curr_local_raw.shape[1] > 0
            ):
                if scale_mode in {"median_all", "trimmed_mean_all"}:
                    actual_overlap = min(overlap_size, prev_local_raw.shape[1] - prev_idx, curr_local_raw.shape[1])
                    prev_depth = prev_local_raw[:, prev_idx : prev_idx + actual_overlap, ..., 2]
                    curr_depth = curr_local_raw[:, :actual_overlap, ..., 2]
                else:
                    prev_depth = prev_local_raw[:, prev_idx, ..., 2]
                    curr_depth = curr_local_raw[:, 0, ..., 2]

                prev_depth_f32 = prev_depth.to(torch.float32)
                curr_depth_f32 = curr_depth.to(torch.float32)
                eps_depth = torch.finfo(torch.float32).eps
                valid = (
                    torch.isfinite(prev_depth_f32)
                    & torch.isfinite(curr_depth_f32)
                    & (curr_depth_f32.abs() > eps_depth)
                )

                prev_depth_flat = prev_depth_f32.reshape(batch_size, -1)
                curr_depth_flat = curr_depth_f32.reshape(batch_size, -1)
                valid_flat = valid.reshape(batch_size, -1)

                scale_values = []
                for batch_idx in range(batch_size):
                    valid_idx = valid_flat[batch_idx]
                    if valid_idx.any():
                        ratios = prev_depth_flat[batch_idx, valid_idx] / curr_depth_flat[batch_idx, valid_idx]
                        if scale_mode in {"median", "median_all", "sim3_avg1"}:
                            scale_val = ratios.median()
                        elif scale_mode in {"trimmed_mean", "trimmed_mean_all"}:
                            scale_val = robust_scale_estimation(ratios.unsqueeze(0), trim_ratio=0.25).squeeze(0)
                        else:
                            raise ValueError(f"Unknown scale_mode: {scale_mode}")
                        scale_values.append(scale_val)
                    else:
                        scale_values.append(torch.tensor(1.0, device=device, dtype=torch.float32))

                relative_scale = torch.stack(scale_values).to(dtype)
                if scale_mode == "sim3_avg1":
                    relative_scale = (relative_scale + 1.0) / 2.0
                relative_scale = torch.clamp(relative_scale, min=1e-3, max=1e3)

        rotated_curr_centers = torch.matmul(relative_rot, t_curr.unsqueeze(-1)).squeeze(-1)
        relative_trans = t_prev - relative_scale.unsqueeze(-1) * rotated_curr_centers
        return relative_scale, relative_rot.to(dtype), relative_trans.to(dtype)

    block_scale: Optional[torch.Tensor] = None
    block_rot: Optional[torch.Tensor] = None
    block_trans: Optional[torch.Tensor] = None

    for window_idx, pred in enumerate(all_predictions):
        if window_idx == 0:
            current_scale = torch.ones_like(one_scale)
            current_rot = identity_rot.clone()
            current_trans = zero_trans.clone()
            if reuse_transform_within_reset_block and reset_every > 0:
                block_scale = current_scale.clone()
                block_rot = current_rot.clone()
                block_trans = current_trans.clone()
        else:
            prev_aligned = aligned_predictions[-1]
            reuse_block_transform = (
                reuse_transform_within_reset_block
                and reset_every > 0
                and window_idx % reset_every != 0
                and block_rot is not None
                and block_trans is not None
            )
            if reuse_block_transform:
                current_rot = block_rot.clone()
                current_trans = block_trans.clone()
                current_scale = block_scale.clone() if allow_scale and block_scale is not None else torch.ones_like(one_scale)
            else:
                current_scale, current_rot, current_trans = estimate_relative_transform(prev_aligned, pred)
                if reuse_transform_within_reset_block and reset_every > 0:
                    block_scale = current_scale.clone()
                    block_rot = current_rot.clone()
                    block_trans = current_trans.clone()

        if allow_scale and sim3_scales is not None:
            sim3_scales.append(current_scale.clone())

        pose_mat = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        pose_mat[:, :3, :3] = current_rot
        pose_mat[:, :3, 3] = current_trans
        sim3_poses.append(pose_mat)

        aligned_pred: PredictionDict = {}
        original_local_points = pred.get("local_points", None)
        aligned_pred["_local_points_raw"] = original_local_points

        if original_local_points is not None:
            scale_factor = current_scale.view(batch_size, 1, 1, 1, 1)
            aligned_local_points = original_local_points * scale_factor if allow_scale else original_local_points
        else:
            aligned_local_points = None
        aligned_pred["local_points"] = aligned_local_points

        def transform_camera(cam_tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if cam_tensor is None:
                return None
            frames = cam_tensor.shape[1]
            rot_local = cam_tensor[..., :3, :3]
            trans_local = cam_tensor[..., :3, 3]
            rot_global = torch.matmul(current_rot.unsqueeze(1).expand(-1, frames, -1, -1), rot_local)
            rotated_trans = torch.matmul(
                current_rot.unsqueeze(1).expand(-1, frames, -1, -1),
                trans_local.unsqueeze(-1),
            ).squeeze(-1)
            if allow_scale:
                rotated_trans = rotated_trans * current_scale.view(batch_size, 1, 1)
            trans_global = rotated_trans + current_trans.unsqueeze(1)
            cam_out = cam_tensor.clone()
            cam_out[..., :3, :3] = rot_global
            cam_out[..., :3, 3] = trans_global
            return cam_out

        camera_global = transform_camera(pred.get("camera_poses", None))
        aligned_pred["camera_poses"] = camera_global
        aligned_pred["local_camera_poses"] = transform_camera(pred.get("local_camera_poses", None))

        if camera_global is not None and aligned_local_points is not None:
            local_points_for_world = aligned_local_points.to(camera_global.dtype)
            aligned_points = torch.einsum(
                "bnij, bnhwj -> bnhwi",
                camera_global,
                homogenize_points(local_points_for_world),
            )[..., :3]
        else:
            points = pred.get("points", None)
            if points is not None:
                aligned_points = torch.einsum("bij, bnhwj -> bnhwi", current_rot, points.to(current_rot.dtype))
                if allow_scale:
                    aligned_points = aligned_points * current_scale.view(batch_size, 1, 1, 1, 1)
                aligned_points = aligned_points + current_trans.view(batch_size, 1, 1, 1, 3)
            else:
                aligned_points = None
        aligned_pred["points"] = aligned_points
        aligned_pred["conf"] = pred.get("conf", None)

        for key, value in pred.items():
            if key in aligned_pred:
                continue
            aligned_pred[key] = value

        aligned_predictions.append(aligned_pred)

    cleaned_predictions: List[PredictionDict] = []
    for pred in aligned_predictions:
        cleaned = dict(pred)
        cleaned.pop("_local_points_raw", None)
        cleaned_predictions.append(cleaned)

    merged = merge_windowed_predictions(cleaned_predictions, window_size, overlap_size)
    pose_key = "chunk_sim3_poses" if allow_scale else "chunk_se3_poses"
    if allow_scale and sim3_scales:
        merged["chunk_sim3_scales"] = torch.stack(sim3_scales, dim=1)
    if sim3_poses:
        merged[pose_key] = torch.stack(sim3_poses, dim=1)
    merged["alignment_mode"] = "sim3" if allow_scale else "se3"
    return merged
