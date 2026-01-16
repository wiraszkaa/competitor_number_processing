"""
Bib image preprocessing (plan step 2.2).

This module provides a small, self-contained preprocessing pipeline for images.
It is designed to be used before uploading or before later bib/number detection steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


@dataclass(frozen=True)
class PreprocessConfig:
    # Scale image so that its longest edge is at most this value (keeps aspect ratio).
    max_long_edge: int = 1280

    # Contrast/illumination normalization
    autocontrast: bool = True
    equalize: bool = False
    gamma: float = 1.0  # 1.0 = no change

    # Noise reduction / detail
    median_filter_size: int = 3  # 0 disables
    gaussian_blur_radius: float = 0.0  # 0 disables
    sharpness: float = 1.0  # 1.0 = no change

    # Global adjustments
    contrast: float = 1.0  # 1.0 = no change


def _resize_keep_aspect(img: Image.Image, max_long_edge: int) -> Image.Image:
    if max_long_edge <= 0:
        return img

    w, h = img.size
    long_edge = max(w, h)
    if long_edge <= max_long_edge:
        return img

    scale = max_long_edge / float(long_edge)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)


def _apply_gamma(img: Image.Image, gamma: float) -> Image.Image:
    # gamma < 1 brightens; gamma > 1 darkens
    if gamma is None or abs(gamma - 1.0) < 1e-6:
        return img
    if gamma <= 0:
        return img

    inv = 1.0 / gamma
    table = [int(pow(i / 255.0, inv) * 255.0) for i in range(256)]
    return img.point(table)


def preprocess_image(
    input_path: Path,
    cfg: PreprocessConfig,
    save_debug_to: Optional[Path] = None,
    prefix: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Preprocess an image and optionally save debug stages.

    Returns a dict with keys: rgb, gray, norm, denoise, final (paths).
    If save_debug_to is None, only the 'final' path is returned and it points to a temporary
    file next to input (with suffix __final.png).
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    stem = prefix or input_path.stem
    out_dir = Path(save_debug_to) if save_debug_to else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}

    # 1) Load and standardize to RGB
    rgb = Image.open(input_path).convert("RGB")
    rgb = _resize_keep_aspect(rgb, cfg.max_long_edge)
    rgb_path = out_dir / f"{stem}__rgb.png"
    rgb.save(rgb_path)
    paths["rgb"] = rgb_path

    # 2) Grayscale
    gray = rgb.convert("L")
    gray_path = out_dir / f"{stem}__gray.png"
    gray.save(gray_path)
    paths["gray"] = gray_path

    # 3) Normalize contrast/illumination
    norm = gray
    if cfg.autocontrast:
        norm = ImageOps.autocontrast(norm)
    if cfg.equalize:
        norm = ImageOps.equalize(norm)

    if cfg.gamma and abs(cfg.gamma - 1.0) > 1e-6:
        norm = _apply_gamma(norm, cfg.gamma)

    if cfg.contrast and abs(cfg.contrast - 1.0) > 1e-6:
        norm = ImageEnhance.Contrast(norm).enhance(cfg.contrast)

    norm_path = out_dir / f"{stem}__norm.png"
    norm.save(norm_path)
    paths["norm"] = norm_path

    # 4) Denoise
    denoise = norm
    if cfg.median_filter_size and cfg.median_filter_size > 0:
        denoise = denoise.filter(ImageFilter.MedianFilter(size=int(cfg.median_filter_size)))
    if cfg.gaussian_blur_radius and cfg.gaussian_blur_radius > 0:
        denoise = denoise.filter(ImageFilter.GaussianBlur(radius=float(cfg.gaussian_blur_radius)))

    denoise_path = out_dir / f"{stem}__denoise.png"
    denoise.save(denoise_path)
    paths["denoise"] = denoise_path

    # 5) Final (optional sharpening)
    final = denoise
    if cfg.sharpness and abs(cfg.sharpness - 1.0) > 1e-6:
        final = ImageEnhance.Sharpness(final).enhance(cfg.sharpness)

    final_path = out_dir / f"{stem}__final.png"
    final.save(final_path)
    paths["final"] = final_path

    return paths
