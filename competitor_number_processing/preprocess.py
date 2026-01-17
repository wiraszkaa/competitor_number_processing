"""
Competitor number image preprocessing.

This module provides a small, self-contained preprocessing pipeline for images.
It is designed to be used before uploading or before later competitor number detection steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


@dataclass(frozen=True)
class PreprocessConfig:
    # Spec 2.2 requirement: Image size and aspect ratio standardization
    # Scale image so that its longest edge is at most this value (keeps aspect ratio).
    max_long_edge: int = 1280

    # Spec 2.2 requirement: Contrast normalization
    autocontrast: bool = True
    equalize: bool = False
    contrast: float = 1.0  # 1.0 = no change

    # Spec 2.2 requirement: Brightness correction or gamma correction
    gamma: float = 1.0  # 1.0 = no change
    brightness: float = 1.0  # 1.0 = no change

    # Spec 2.2 requirement: Noise reduction
    median_filter_size: int = 3  # 0 disables
    gaussian_blur_radius: float = 0.0  # 0 disables


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
    if abs(gamma - 1.0) < 1e-6:
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
    Preprocess an image according to specification 2.2.

    Pipeline (spec 2.2):
    1. Size and aspect ratio standardization
    2. Brightness correction or gamma correction
    3. Contrast normalization
    4. Noise reduction

    Returns a dict with keys: resized, brightness, contrast, denoise, final (paths).
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

    # Step 1: Load as RGB and standardize size/aspect ratio (spec 2.2)
    img = Image.open(input_path).convert("RGB")
    img = _resize_keep_aspect(img, cfg.max_long_edge)
    resized_path = out_dir / f"{stem}__resized.png"
    img.save(resized_path)
    paths["resized"] = resized_path

    # Step 2: Brightness correction or gamma correction (spec 2.2)
    if cfg.gamma and abs(cfg.gamma - 1.0) > 1e-6:
        img = _apply_gamma(img, cfg.gamma)

    if cfg.brightness and abs(cfg.brightness - 1.0) > 1e-6:
        img = ImageEnhance.Brightness(img).enhance(cfg.brightness)

    brightness_path = out_dir / f"{stem}__brightness.png"
    img.save(brightness_path)
    paths["brightness"] = brightness_path

    # Step 3: Contrast normalization (spec 2.2)
    if cfg.autocontrast:
        img = ImageOps.autocontrast(img)
    if cfg.equalize:
        img = ImageOps.equalize(img)
    if cfg.contrast and abs(cfg.contrast - 1.0) > 1e-6:
        img = ImageEnhance.Contrast(img).enhance(cfg.contrast)

    contrast_path = out_dir / f"{stem}__contrast.png"
    img.save(contrast_path)
    paths["contrast"] = contrast_path

    # Step 4: Noise reduction (spec 2.2)
    if cfg.median_filter_size and cfg.median_filter_size > 0:
        img = img.filter(ImageFilter.MedianFilter(size=int(cfg.median_filter_size)))
    if cfg.gaussian_blur_radius and cfg.gaussian_blur_radius > 0:
        img = img.filter(
            ImageFilter.GaussianBlur(radius=float(cfg.gaussian_blur_radius))
        )

    denoise_path = out_dir / f"{stem}__denoise.png"
    img.save(denoise_path)
    paths["denoise"] = denoise_path

    # Final output
    final_path = out_dir / f"{stem}__final.png"
    img.save(final_path)
    paths["final"] = final_path

    return paths
