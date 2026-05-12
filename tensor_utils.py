from __future__ import annotations

import glob
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from natsort import natsorted
from torchvision.io import ImageReadMode, read_image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def list_image_files(input_dir: str | Path) -> List[str]:
    paths: List[str] = []
    base_dir = Path(input_dir)
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        paths.extend(glob.glob(str(base_dir / pattern)))
    image_paths = natsorted(paths)
    return [path for path in image_paths if "depth" not in Path(path).name.lower()]


def _decode_image(path: str) -> torch.Tensor:
    return read_image(path, mode=ImageReadMode.RGB)


def _infer_target_resolution(
    first_image: torch.Tensor,
    *,
    pixel_limit: int,
) -> Tuple[int, int]:
    height = int(first_image.shape[-2])
    width = int(first_image.shape[-1])
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
    return target_width, target_height


def _resize_batch(batch: torch.Tensor, *, target_width: int, target_height: int) -> torch.Tensor:
    if batch.ndim != 4:
        raise ValueError(f"Expected batch tensor with shape [N, C, H, W], got {tuple(batch.shape)}")
    if int(batch.shape[-1]) == target_width and int(batch.shape[-2]) == target_height:
        return batch
    return F.interpolate(
        batch,
        size=(target_height, target_width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )


def load_images_from_paths(
    image_paths: Sequence[str],
    *,
    pixel_limit: int = 255000,
    target_resolution: Optional[Tuple[int, int]] = None,
    verbose: bool = True,
    device: Optional[torch.device] = None,
    decode_workers: Optional[int] = None,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not image_paths:
        return torch.empty(0)

    if decode_workers is None:
        cpu_count = os.cpu_count() or 1
        decode_workers = max(1, min(8, cpu_count))
    else:
        decode_workers = max(1, int(decode_workers))

    with ThreadPoolExecutor(max_workers=decode_workers) as pool:
        decoded_images = list(pool.map(_decode_image, image_paths))

    if target_resolution is None:
        target_width, target_height = _infer_target_resolution(decoded_images[0], pixel_limit=pixel_limit)
    else:
        target_width, target_height = int(target_resolution[0]), int(target_resolution[1])

    if verbose:
        print(f"All images will be resized to a uniform size: ({target_width}, {target_height})")

    same_shape = all(tuple(img.shape) == tuple(decoded_images[0].shape) for img in decoded_images)

    if same_shape:
        batch = torch.stack(decoded_images, dim=0)
        if device.type == "cuda":
            batch = batch.pin_memory()
        batch = batch.to(device, non_blocking=True).to(torch.float32).div_(255.0)
        batch = _resize_batch(batch, target_width=target_width, target_height=target_height)
        return batch.clamp_(0.0, 1.0).contiguous()

    resized_images: List[torch.Tensor] = []
    for image in decoded_images:
        tensor = image.to(device, non_blocking=True).to(torch.float32).div_(255.0).unsqueeze(0)
        tensor = _resize_batch(tensor, target_width=target_width, target_height=target_height)
        resized_images.append(tensor.clamp_(0.0, 1.0).squeeze(0).contiguous())
    return torch.stack(resized_images, dim=0)


__all__ = ["list_image_files", "load_images_from_paths"]
