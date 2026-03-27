import argparse

from data_utils import load_result_for_viser, load_result_meta
from loger.utils.viser_utils import viser_wrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a saved result directory with viser.")
    parser.add_argument("--result_dir", "--annotation_dir", dest="result_dir", type=str, required=True, help="Result directory created by annote_dataset.py.")
    parser.add_argument("--port", type=int, default=8080, help="Port number for the viser server.")
    parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode.")
    parser.add_argument("--share", action="store_true", help="Share the viser server with others.")
    parser.add_argument("--conf_threshold", type=float, default=20.0, help="Initial confidence threshold.")
    parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation if onnxruntime is available.")
    parser.add_argument("--subsample", type=int, default=5, help="Point cloud subsample factor.")
    parser.add_argument("--video_width", type=int, default=320, help="Video preview width in the GUI.")
    parser.add_argument("--window_size", type=int, default=None, help="Override the sliding window size used for visualization.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = load_result_meta(args.result_dir)
    pred_dict = load_result_for_viser(args.result_dir)
    if args.window_size is not None:
        pred_dict["window_size"] = args.window_size
    viser_wrapper(
        pred_dict,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        background_mode=args.background_mode,
        mask_sky=args.mask_sky,
        image_folder_for_sky_mask=meta.get("frame_dir"),
        subsample=args.subsample,
        video_width=args.video_width,
        share=args.share,
        canonical_first_frame=True,
    )


if __name__ == "__main__":
    main()
