import argparse
from pathlib import Path

from data_utils import load_result_for_viser
from loger.utils.viser_utils import viser_wrapper
from align_ground import apply_transform_to_points, apply_transform_to_poses, estimate_trajectory_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a saved result directory with viser.")
    parser.add_argument("--result_dir", dest="result_dir", type=str, required=True, help="Result directory created by annote_dataset.py.")
    parser.add_argument("--frame_dir", type=str, default=None, help="Directory containing the RGB frames for this result. If omitted, infer it from result_dir by replacing /traj/ with /img/.")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame index for segmented visualization.")
    parser.add_argument("--end_frame", type=int, default=-1, help="End frame index (exclusive). Use -1 for the last frame.")
    
    parser.add_argument("--port", type=int, default=8080, help="Port number for the viser server.")
    parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode.")
    parser.add_argument("--share", action="store_true", help="Share the viser server with others.")
    parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation if onnxruntime is available.")
    
    parser.add_argument("--conf_threshold", type=float, default=20.0, help="Initial confidence threshold.")
    parser.add_argument("--subsample", type=int, default=5, help="Point cloud subsample factor.")
    parser.add_argument("--video_width", type=int, default=320, help="Video preview width in the GUI.")
    parser.add_argument(
        "--reference_frame",
        type=str,
        default="auto",
        choices=["auto", "initial_camera", "result", "trajectory_plane"],
        help="Which reference frame to visualize in.",
    )
    parser.add_argument("--frame_step", type=int, default=8, help="Use the first N frames to estimate up, and the displacement from frame 0 to frame N to define forward in the trajectory-aligned frame.")
    return parser.parse_args()


def infer_frame_dir(result_dir: Path) -> Path:
    parts = list(result_dir.resolve().parts)
    if "traj" not in parts:
        raise ValueError("Could not infer frame_dir automatically because result_dir does not contain a 'traj' path component.")
    traj_idx = parts.index("traj")
    inferred_parts = parts[:]
    inferred_parts[traj_idx] = "img"
    return Path(*inferred_parts)

def _resolve_reference_frame(args: argparse.Namespace, pred_dict: dict) -> str:
    if args.reference_frame != "auto":
        return args.reference_frame
    if pred_dict.get("use_result_frame", False):
        return "result"
    return "initial_camera"


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir).expanduser().resolve()
    frame_dir = Path(args.frame_dir).expanduser().resolve() if args.frame_dir is not None else infer_frame_dir(result_dir)
    pred_dict = load_result_for_viser(
        result_dir,
        frame_dir,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    pred_dict["sequence_name"] = result_dir.name
    reference_frame = _resolve_reference_frame(args, pred_dict)
    if reference_frame == "trajectory_plane":
        alignment = estimate_trajectory_frame(
            pred_dict["camera_poses"],
            frame_step=args.frame_step,
            plane_origin="first_camera",
        )
        pred_dict["points"] = apply_transform_to_points(pred_dict["points"], alignment["transform"])
        pred_dict["camera_poses"] = apply_transform_to_poses(pred_dict["camera_poses"], alignment["transform"])
    canonical_first_frame = reference_frame == "initial_camera"
    viser_wrapper(
        pred_dict,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        background_mode=args.background_mode,
        mask_sky=args.mask_sky,
        image_folder_for_sky_mask=str(frame_dir),
        subsample=args.subsample,
        video_width=args.video_width,
        share=args.share,
        canonical_first_frame=canonical_first_frame,
    )


if __name__ == "__main__":
    main()
