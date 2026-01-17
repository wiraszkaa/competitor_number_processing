"""
Competitor number image preprocessing.

This module provides a small, self-contained preprocessing pipeline for images.
It is designed to be used before uploading or before later competitor number detection steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple

import cv2
import numpy as np
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

    # Grass-aware preprocessing for sports images (Spec 2.3)
    # Enhances people detection on grass backgrounds
    enable_grass_preprocessing: bool = False
    grass_edge_enhancement: bool = True  # Add strong edges between grass and people
    grass_sharpening: bool = True  # Sharpen non-grass regions


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


def _detect_grass_color(image_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect the dominant grass color in the image dynamically.

    Args:
        image_np: Input image as numpy array in BGR format

    Returns:
        Tuple of (lower_bound, upper_bound) in HSV color space
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)

    # Focus on green hues (30-90 in HSV)
    green_mask = cv2.inRange(hsv, np.array([30, 30, 30]), np.array([90, 255, 255]))

    # Get pixels that are greenish
    green_pixels = hsv[green_mask > 0]

    if len(green_pixels) == 0:
        # No green found, use default grass color
        return np.array([35, 40, 40]), np.array([85, 255, 255])

    # Calculate median of green pixels (more robust than mean)
    median_h = int(np.median(green_pixels[:, 0]))
    median_s = int(np.median(green_pixels[:, 1]))
    median_v = int(np.median(green_pixels[:, 2]))

    # Create wider range around the detected grass color
    h_range = 15  # Hue tolerance
    s_range = 60  # Saturation tolerance
    v_range = 80  # Value tolerance

    lower_bound = np.array(
        [
            max(0, median_h - h_range),
            max(0, median_s - s_range),
            max(0, median_v - v_range),
        ]
    )

    upper_bound = np.array([min(179, median_h + h_range), 255, 255])

    return lower_bound, upper_bound


def _apply_grass_preprocessing(
    img: Image.Image,
    enable_edge_enhancement: bool = True,
    enable_sharpening: bool = True,
) -> Image.Image:
    """
    Apply grass-aware preprocessing to enhance people detection on grass backgrounds.

    This preprocessing:
    1. Dynamically detects the grass color in the image
    2. Creates strong contours between grass and non-grass regions
    3. Enhances contrast in non-grass areas (likely people)
    4. Sharpens non-grass regions for better feature detection

    Args:
        img: Input PIL Image
        enable_edge_enhancement: Add strong edges between grass and people
        enable_sharpening: Sharpen non-grass regions

    Returns:
        Preprocessed PIL Image
    """
    # Convert PIL to numpy array (BGR for OpenCV)
    img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Detect dominant grass color dynamically
    lower_grass, upper_grass = _detect_grass_color(img_np)

    # Convert to HSV for color segmentation
    hsv = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV)

    # Create mask for detected grass color
    grass_mask = cv2.inRange(hsv, lower_grass, upper_grass)

    # Apply aggressive morphological operations to fill grass regions
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    grass_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_CLOSE, kernel_large)
    grass_mask = cv2.dilate(grass_mask, kernel_large, iterations=1)

    # Invert to get non-grass areas (people)
    non_grass_mask = cv2.bitwise_not(grass_mask)

    # Clean up the non-grass mask
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    non_grass_mask = cv2.morphologyEx(non_grass_mask, cv2.MORPH_OPEN, kernel_small)

    # Enhance contrast in non-grass regions
    lab = cv2.cvtColor(img_np, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Aggressive CLAHE on non-grass areas
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # Apply enhancement only to non-grass areas
    l_result = np.where(non_grass_mask > 0, l_enhanced, l)

    enhanced = cv2.merge([l_result, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Add edge enhancement if enabled
    if enable_edge_enhancement:
        # Find edges between grass and non-grass
        edges = cv2.Canny(grass_mask, 50, 150)
        edges_dilated = cv2.dilate(edges, kernel_small, iterations=2)

        # Add edges to the image
        edge_image = cv2.cvtColor(edges_dilated, cv2.COLOR_GRAY2BGR)
        enhanced = cv2.add(enhanced, edge_image)

    # Apply sharpening to non-grass areas if enabled
    if enable_sharpening:
        kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)

        # Apply sharpening only to non-grass areas
        enhanced = np.where(non_grass_mask[:, :, np.newaxis] > 0, sharpened, enhanced)

    # Convert back to PIL Image
    result = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result)


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

    # Final output WITHOUT grass enhancement (for OCR/number detection)
    final_path = out_dir / f"{stem}__final.png"
    img.save(final_path)
    paths["final"] = final_path

    # Step 5: Grass-aware preprocessing (spec 2.3 - for sports images)
    # This step enhances people detection on grass backgrounds
    # Save separately as it makes numbers harder to detect
    if cfg.enable_grass_preprocessing:
        img_grass = _apply_grass_preprocessing(
            img,
            enable_edge_enhancement=cfg.grass_edge_enhancement,
            enable_sharpening=cfg.grass_sharpening,
        )

        grass_path = out_dir / f"{stem}__grass_enhanced.png"
        img_grass.save(grass_path)
        paths["grass_enhanced"] = grass_path

    return paths
