from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import yaml

from loger.models.pi3 import Pi3
from loger.models.pi3x import Pi3X
from loger.window_inference import (
    compute_windows,
    merge_windowed_predictions,
    merge_windowed_predictions_sim3,
)


def _read_config_file(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _maybe_parse_sequence(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = yaml.safe_load(stripped)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
            except Exception:
                pass
    return value


def _is_local_pi3x_dir(model_name: str) -> bool:
    model_dir = Path(model_name)
    if not model_dir.is_dir():
        return False
    if not (model_dir / "config.json").is_file():
        return False
    return any((model_dir / candidate).is_file() for candidate in ("model.safetensors", "pytorch_model.bin"))


def load_model_init_kwargs(config_path: Optional[str]) -> Dict[str, Any]:
    if config_path is None or not Path(config_path).is_file():
        return {}

    config = _read_config_file(config_path)
    model_config = config.get("model", {})
    pi3_signature = inspect.signature(Pi3.__init__)
    valid_kwargs = {
        name
        for name, param in pi3_signature.parameters.items()
        if name not in {"self", "args", "kwargs"}
        and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }

    model_kwargs: Dict[str, Any] = {}
    for key in sorted(valid_kwargs):
        if key in model_config:
            value = model_config[key]
            if key in {"ttt_insert_after", "attn_insert_after"}:
                value = _maybe_parse_sequence(value)
            model_kwargs[key] = value
    return model_kwargs


def load_forward_defaults(config_path: Optional[str]) -> Dict[str, Any]:
    if config_path is None or not Path(config_path).is_file():
        return {}

    config = _read_config_file(config_path)
    training_settings = config.get("training_settings", {})
    model_settings = config.get("model", {})
    return {
        "window_size": training_settings.get("window_size", 32),
        "overlap_size": training_settings.get("overlap_size", 3),
        "reset_every": training_settings.get("reset_every", 0),
        "num_iterations": config.get("num_iterations", 1),
        "sim3": config.get("sim3", False),
        "sim3_scale_mode": config.get("sim3_scale_mode", "median"),
        "se3": bool(model_settings.get("se3", config.get("se3", False))),
    }


def load_reconstruction_model(
    model_name: str,
    *,
    model_config_path: Optional[str] = None,
    pi3x: bool = True,
    pi3x_metric: bool = True,
    use_multimodal: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[torch.nn.Module, str]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if pi3x and _is_local_pi3x_dir(model_name):
        model = Pi3X.from_pretrained(model_name, use_multimodal=use_multimodal)
        model = model.to(device).eval()
        return model, "native_pi3x"

    checkpoint_path = Path(model_name)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Expected a local Pi3X directory or local LoGeR checkpoint, but got: {model_name}"
        )

    model_kwargs = load_model_init_kwargs(model_config_path)
    if pi3x:
        model_kwargs["pi3x"] = True
        model_kwargs["pi3x_metric"] = pi3x_metric
    model = Pi3(**model_kwargs)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    cleaned = {key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()}
    model.load_state_dict(cleaned, strict=False if pi3x else True)

    model = model.to(device).eval()
    return model, "loger_pi3"


def _get_autocast_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8:
        return torch.bfloat16
    return torch.float16


def _to_cpu_predictions(raw_outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    keep_keys = {"local_points", "conf", "camera_poses", "metric"}
    predictions: Dict[str, torch.Tensor] = {}
    for key, value in raw_outputs.items():
        if key in keep_keys and torch.is_tensor(value):
            tensor = value.detach().cpu()
            if key in {"local_points", "conf", "metric"}:
                tensor = tensor.to(torch.float16)
            else:
                tensor = tensor.to(torch.float32)
            predictions[key] = tensor
    return predictions


def run_native_pi3x_windowed_inference(
    model: torch.nn.Module,
    images_tensor: torch.Tensor,
    *,
    device: torch.device,
    window_size: int,
    overlap_size: int,
    sim3: bool,
    se3: bool,
    reset_every: int,
    sim3_scale_mode: str,
    show_progress: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    num_frames = int(images_tensor.shape[0])
    windows, eff_window_size, eff_overlap = compute_windows(num_frames, window_size, overlap_size)
    dtype = _get_autocast_dtype(device)
    all_predictions: List[Dict[str, torch.Tensor]] = []

    if show_progress:
        print(f"Running native Pi3X windowed inference with {len(windows)} window(s).")

    start_time = time.time()
    for window_idx, (start_idx, end_idx) in enumerate(windows, start=1):
        if show_progress:
            print(f"  Window {window_idx}/{len(windows)}: frames [{start_idx}, {end_idx})")
        window_tensor = images_tensor[start_idx:end_idx].to(device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda", dtype=dtype):
            raw_outputs = model(window_tensor.unsqueeze(0))
        raw_outputs["conf"] = torch.sigmoid(raw_outputs["conf"])
        all_predictions.append(_to_cpu_predictions(raw_outputs))
        del raw_outputs, window_tensor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    align_on_resets_without_explicit_pose = reset_every > 0 and not sim3 and not se3
    if sim3:
        merged = merge_windowed_predictions_sim3(
            all_predictions,
            eff_window_size,
            eff_overlap,
            allow_scale=True,
            scale_mode=sim3_scale_mode,
        )
    elif se3 or align_on_resets_without_explicit_pose:
        merged = merge_windowed_predictions_sim3(
            all_predictions,
            eff_window_size,
            eff_overlap,
            allow_scale=False,
            reset_every=reset_every,
            reuse_transform_within_reset_block=align_on_resets_without_explicit_pose,
        )
    else:
        merged = merge_windowed_predictions(all_predictions, eff_window_size, eff_overlap)

    stats = {
        "num_frames": num_frames,
        "num_windows": len(windows),
        "effective_window_size": eff_window_size,
        "effective_overlap_size": eff_overlap,
        "inference_seconds": round(time.time() - start_time, 3),
    }
    return merged, stats


def run_inference(
    model: torch.nn.Module,
    model_kind: str,
    images_tensor: torch.Tensor,
    *,
    device: torch.device,
    forward_kwargs: Dict[str, Any],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    if images_tensor.ndim != 4:
        raise ValueError(f"Expected images_tensor with shape [N, C, H, W], got {tuple(images_tensor.shape)}")

    num_frames = int(images_tensor.shape[0])
    if num_frames == 0:
        raise RuntimeError("No images were provided for inference.")

    if model_kind == "native_pi3x":
        return run_native_pi3x_windowed_inference(
            model,
            images_tensor,
            device=device,
            window_size=int(forward_kwargs.get("window_size", 32)),
            overlap_size=int(forward_kwargs.get("overlap_size", 3)),
            sim3=bool(forward_kwargs.get("sim3", False)),
            se3=bool(forward_kwargs.get("se3", False)),
            reset_every=int(forward_kwargs.get("reset_every", 0) or 0),
            sim3_scale_mode=str(forward_kwargs.get("sim3_scale_mode", "median")),
        )

    start_time = time.time()
    device_images = images_tensor.to(device)
    dtype = _get_autocast_dtype(device)
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda", dtype=dtype):
        raw_outputs = model(device_images.unsqueeze(0), **forward_kwargs)
    raw_outputs["conf"] = torch.sigmoid(raw_outputs["conf"])
    predictions = _to_cpu_predictions(raw_outputs)
    stats = {
        "num_frames": num_frames,
        "num_windows": 1,
        "effective_window_size": num_frames,
        "effective_overlap_size": 0,
        "inference_seconds": round(time.time() - start_time, 3),
    }
    return predictions, stats


def build_forward_kwargs(
    *,
    config_path: Optional[str],
    window_size: Optional[int],
    overlap_size: Optional[int],
    reset_every: Optional[int],
    sim3: bool,
    se3: Optional[bool],
    sim3_scale_mode: str,
    no_ttt: bool = False,
    no_swa: bool = False,
) -> Dict[str, Any]:
    defaults = load_forward_defaults(config_path)
    kwargs = {
        "window_size": defaults.get("window_size", 32),
        "overlap_size": defaults.get("overlap_size", 3),
        "reset_every": defaults.get("reset_every", 0),
        "num_iterations": defaults.get("num_iterations", 1),
        "sim3": defaults.get("sim3", False),
        "sim3_scale_mode": defaults.get("sim3_scale_mode", sim3_scale_mode),
        "se3": defaults.get("se3", True),
        "turn_off_ttt": no_ttt,
        "turn_off_swa": no_swa,
    }
    if window_size is not None:
        kwargs["window_size"] = window_size
    if overlap_size is not None:
        kwargs["overlap_size"] = overlap_size
    if reset_every is not None:
        kwargs["reset_every"] = reset_every
    if sim3:
        kwargs["sim3"] = True
    if se3 is not None:
        kwargs["se3"] = se3
    if sim3_scale_mode:
        kwargs["sim3_scale_mode"] = sim3_scale_mode
    return kwargs


    
