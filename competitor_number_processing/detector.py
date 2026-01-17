"""
Competitor person detection module.

This module implements person detection using HOG (Histogram of Oriented Gradients)
with SVM (Support Vector Machine) classifier according to specification 2.3.

Main features:
- Detection of people using classical methods (HOG + SVM)
- Contour-based detection as alternative method
- Region extraction for detected competitors
- Bounding box visualization

Note: Preprocessing (including grass-aware preprocessing) should be done using
the preprocess.py module before detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectionConfig:
    """Configuration for person detection."""

    # HOG detector parameters
    # Scale factor for image pyramid - reduces by this factor each iteration
    scale: float = 1.05

    # Minimum number of neighboring detections to consider (reduces false positives)
    min_neighbors: int = 1

    # Minimum detection window size (width, height) in pixels
    # Smaller for sports images where people may be at various distances
    min_size: Tuple[int, int] = (40, 80)

    # Maximum detection window size (width, height) in pixels - None means no limit
    max_size: Optional[Tuple[int, int]] = None

    # Padding around detected regions (in pixels)
    padding: int = 15

    # Confidence threshold for detection (0.0 to 1.0)
    # Lower values = more detections but more false positives
    threshold: float = -0.5

    # Enable contour-based detection as fallback/addition
    use_contour_detection: bool = True

    # Minimum contour area for person detection (in pixels)
    min_contour_area: int = 2000


@dataclass(frozen=True)
class DetectedPerson:
    """Represents a detected person in an image."""

    # Bounding box coordinates (x, y, width, height)
    x: int
    y: int
    width: int
    height: int

    # Detection confidence score
    confidence: float

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Return bounding box as (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)

    @property
    def center(self) -> Tuple[int, int]:
        """Return center point of the detection."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        """Return area of the detection."""
        return self.width * self.height


class PersonDetector:
    """
    Person detector using HOG + SVM for detecting people in images.

    This class uses OpenCV's pre-trained HOG descriptor for people detection,
    enhanced with contour-based detection for sports scenarios.

    Note: For best results on sports images with grass backgrounds, preprocess
    images using PreprocessConfig with enable_grass_preprocessing=True.
    """

    def __init__(self, config: Optional[DetectionConfig] = None):
        """Initialize the person detector."""
        self.config = config or DetectionConfig()

        # Initialize HOG descriptor with default people detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def _detect_grass_dominant_color(
        self, hsv: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect the dominant grass color in the image to better identify non-grass regions.

        Args:
            hsv: Image in HSV color space

        Returns:
            Tuple of (lower_bound, upper_bound) for dominant grass color
        """
        # Focus on green hues (30-90 in HSV)
        green_mask = cv2.inRange(hsv, np.array([30, 30, 30]), np.array([90, 255, 255]))

        # Get pixels that are greenish
        green_pixels = hsv[green_mask > 0]

        if len(green_pixels) == 0:
            # No green found, use default
            return np.array([35, 40, 40]), np.array([85, 255, 255])

        # Calculate median of green pixels (dominant grass color)
        median_h = int(np.median(green_pixels[:, 0]))
        median_s = int(np.median(green_pixels[:, 1]))
        median_v = int(np.median(green_pixels[:, 2]))

        # Create range around dominant grass color (tight range to find differences)
        h_range = 10  # Tighter hue tolerance to detect shade differences
        s_range = 40  # Saturation tolerance
        v_range = 50  # Value tolerance

        lower_bound = np.array(
            [
                max(0, median_h - h_range),
                max(0, median_s - s_range),
                max(0, median_v - v_range),
            ]
        )

        upper_bound = np.array(
            [
                min(179, median_h + h_range),
                min(255, median_s + s_range),
                min(255, median_v + v_range),
            ]
        )

        return lower_bound, upper_bound

    def _detect_by_contours(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Enhanced detection using color difference analysis to find people on grass.

        This method:
        1. Detects the dominant grass color in the image
        2. Finds regions that differ from the grass (different shades/colors)
        3. Uses HOG features to validate human-like structures
        4. Combines edge detection for better boundaries

        Args:
            image: Input image in BGR format

        Returns:
            List of bounding boxes (x, y, width, height)
        """
        h, w = image.shape[:2]

        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Detect dominant grass color
        lower_grass, upper_grass = self._detect_grass_dominant_color(hsv)

        # Create mask for dominant grass color
        grass_mask = cv2.inRange(hsv, lower_grass, upper_grass)

        # Invert to get non-grass regions (people, shadows, different colored areas)
        non_grass_mask = cv2.bitwise_not(grass_mask)

        # Additional color variance detection - find regions that differ even slightly
        # Split into H, S, V channels
        h_channel, s_channel, v_channel = cv2.split(hsv)

        # Detect areas with different saturation or value (even if hue is similar)
        # This catches different shades of green (jerseys, shadows on people)
        median_s = (
            np.median(s_channel[grass_mask > 0]) if np.any(grass_mask > 0) else 128
        )
        median_v = (
            np.median(v_channel[grass_mask > 0]) if np.any(grass_mask > 0) else 128
        )

        # Find regions with significantly different saturation or value
        s_diff = np.abs(s_channel.astype(np.float32) - median_s)
        v_diff = np.abs(v_channel.astype(np.float32) - median_v)

        # Threshold for differences (detect even subtle variations)
        variance_mask = ((s_diff > 30) | (v_diff > 30)).astype(np.uint8) * 255

        # Combine with non-grass mask
        combined_mask = cv2.bitwise_or(non_grass_mask, variance_mask)

        # Edge detection on grayscale for structure boundaries
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Enhanced edge detection
        edges_strong = cv2.Canny(gray, 50, 150)
        edges_weak = cv2.Canny(gray, 20, 60)
        edges = cv2.bitwise_or(edges_strong, edges_weak)

        # Dilate edges slightly to connect nearby edges
        kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel_edge, iterations=1)

        # Combine all masks
        final_mask = cv2.bitwise_or(combined_mask, edges)

        # Morphological operations to clean up and connect regions
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        # Close small gaps
        final_mask = cv2.morphologyEx(
            final_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2
        )

        # Remove small noise
        final_mask = cv2.morphologyEx(
            final_mask, cv2.MORPH_OPEN, kernel_small, iterations=1
        )

        # Find contours
        contours, _ = cv2.findContours(
            final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter and validate contours
        bboxes = []
        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.config.min_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Skip regions at the very edge of the image (likely artifacts)
            if (
                x <= 5
                or y <= 5
                or x + w >= image.shape[1] - 5
                or y + h >= image.shape[0] - 5
            ):
                # Only skip if it's very large (likely full image detection)
                if w > image.shape[1] * 0.8 and h > image.shape[0] * 0.8:
                    continue

            # Check aspect ratio - accept wider range for various poses
            aspect_ratio = h / w if w > 0 else 0

            # Very lenient aspect ratio for sports (running, jumping, diving)
            if 0.5 < aspect_ratio < 8.0:
                if w >= self.config.min_size[0] and h >= self.config.min_size[1]:
                    if self.config.max_size is None or (
                        w <= self.config.max_size[0] and h <= self.config.max_size[1]
                    ):
                        # Additional validation: check if region has structure
                        roi = gray[y : y + h, x : x + w]

                        # Check edge density in ROI (humans have edges)
                        roi_edges = edges[y : y + h, x : x + w]
                        edge_density = np.sum(roi_edges > 0) / (w * h)

                        # If there's some structure (edges), it's likely a person
                        if edge_density > 0.02:  # At least 2% edge pixels
                            bboxes.append((x, y, w, h))

        return bboxes

    def _expand_bbox_by_color_similarity(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        color_threshold: int = 40,
    ) -> Tuple[int, int, int, int]:
        """
        Expand bounding box to include connected regions with similar colors.
        Only expands moderately to avoid over-expansion.

        Args:
            image: Input image in BGR format
            x, y, w, h: Initial bounding box
            color_threshold: Maximum color difference to consider similar

        Returns:
            Expanded bounding box (x, y, w, h)
        """
        img_h, img_w = image.shape[:2]

        # Limit expansion to reasonable amount
        max_expansion_factor = 1.5  # Max 50% expansion in each direction

        # Get median color of the initial region
        roi = image[max(0, y) : min(img_h, y + h), max(0, x) : min(img_w, x + w)]
        if roi.size == 0:
            return x, y, w, h

        median_color = np.median(roi.reshape(-1, 3), axis=0)

        # Expand in each direction gradually
        margin = 15
        search_x1 = max(0, x - int(w * 0.3))
        search_y1 = max(0, y - int(h * 0.3))
        search_x2 = min(img_w, x + w + int(w * 0.3))
        search_y2 = min(img_h, y + h + int(h * 0.3))

        # Get the search region
        search_region = image[search_y1:search_y2, search_x1:search_x2]

        # Find similar colors in search region
        color_diff = np.sqrt(
            np.sum((search_region.astype(np.float32) - median_color) ** 2, axis=2)
        )
        similar_mask = (color_diff < color_threshold).astype(np.uint8) * 255

        # Find contours in similar region
        contours, _ = cv2.findContours(
            similar_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return x, y, w, h

        # Find the contour that contains the original bbox center
        center_x_local = x + w // 2 - search_x1
        center_y_local = y + h // 2 - search_y1

        for contour in contours:
            if (
                cv2.pointPolygonTest(contour, (center_x_local, center_y_local), False)
                >= 0
            ):
                # Get bounding box of this contour
                cx, cy, cw, ch = cv2.boundingRect(contour)

                # Convert back to image coordinates
                exp_x = search_x1 + cx
                exp_y = search_y1 + cy
                exp_w = cw
                exp_h = ch

                # Limit expansion
                max_w = int(w * max_expansion_factor)
                max_h = int(h * max_expansion_factor)

                if exp_w > max_w or exp_h > max_h:
                    # Keep original if expansion is too large
                    return x, y, w, h

                return exp_x, exp_y, exp_w, exp_h

        return x, y, w, h

    def _merge_overlapping_boxes(
        self, detections: List[DetectedPerson], iou_threshold: float = 0.3
    ) -> List[DetectedPerson]:
        """
        Merge bounding boxes that are overlapping or nested.

        Args:
            detections: List of detected persons
            iou_threshold: IoU threshold for merging (also merges if one box is inside another)

        Returns:
            List of merged detections
        """
        if len(detections) <= 1:
            return detections

        # Convert to list for modification
        boxes = list(detections)
        merged = []

        while boxes:
            current = boxes.pop(0)
            to_merge = [current]
            remaining = []

            for other in boxes:
                # Calculate IoU
                x1 = max(current.x, other.x)
                y1 = max(current.y, other.y)
                x2 = min(current.x + current.width, other.x + other.width)
                y2 = min(current.y + current.height, other.y + other.height)

                if x1 < x2 and y1 < y2:
                    intersection = (x2 - x1) * (y2 - y1)
                    area_current = current.width * current.height
                    area_other = other.width * other.height

                    # Check if one box is inside another
                    is_inside = (intersection / min(area_current, area_other)) > 0.7

                    # Check standard IoU
                    iou = intersection / (area_current + area_other - intersection)

                    if iou > iou_threshold or is_inside:
                        to_merge.append(other)
                    else:
                        remaining.append(other)
                else:
                    remaining.append(other)

            # Merge all overlapping boxes
            if to_merge:
                x_min = min(d.x for d in to_merge)
                y_min = min(d.y for d in to_merge)
                x_max = max(d.x + d.width for d in to_merge)
                y_max = max(d.y + d.height for d in to_merge)

                # Use max confidence
                max_conf = max(d.confidence for d in to_merge)

                merged.append(
                    DetectedPerson(
                        x=x_min,
                        y=y_min,
                        width=x_max - x_min,
                        height=y_max - y_min,
                        confidence=max_conf,
                    )
                )

            boxes = remaining

        return merged

    def detect(self, image: np.ndarray) -> List[DetectedPerson]:
        """
        Detect people in an image using HOG + SVM.

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            List of DetectedPerson objects
        """
        # Ensure image is in correct format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        # Perform HOG + SVM detection
        detections_raw, weights = self.hog.detectMultiScale(
            image,
            winStride=(4, 4),
            padding=(16, 16),
            scale=self.config.scale,
            useMeanshiftGrouping=False,
        )

        # Apply non-maximum suppression
        bboxes = []
        filtered_weights = []

        if len(detections_raw) > 0:
            grouped_rects, grouped_weights = cv2.groupRectangles(
                detections_raw.tolist(), self.config.min_neighbors, eps=0.2
            )

            if len(grouped_rects) > 0:
                bboxes = np.array(grouped_rects)
                filtered_weights = (
                    weights[: len(grouped_rects)]
                    if len(weights) > 0
                    else np.ones(len(grouped_rects))
                )
            else:
                bboxes = np.array([])
                filtered_weights = np.array([])
        else:
            bboxes = np.array([])
            filtered_weights = np.array([])

        weights = filtered_weights

        # Process HOG detections
        detections = []

        if len(bboxes) > 0:
            for i, (x, y, w, h) in enumerate(bboxes):
                if w < self.config.min_size[0] or h < self.config.min_size[1]:
                    continue

                if self.config.max_size is not None:
                    if w > self.config.max_size[0] or h > self.config.max_size[1]:
                        continue

                weight = float(weights[i]) if i < len(weights) else 1.0

                if weight < self.config.threshold:
                    continue

                detections.append(
                    DetectedPerson(
                        x=int(x),
                        y=int(y),
                        width=int(w),
                        height=int(h),
                        confidence=weight,
                    )
                )

        # Add contour-based detections if enabled
        if self.config.use_contour_detection:
            contour_bboxes = self._detect_by_contours(image)

            for x, y, w, h in contour_bboxes:
                # Check if overlaps with existing detection
                is_duplicate = False
                for existing in detections:
                    x1 = max(x, existing.x)
                    y1 = max(y, existing.y)
                    x2 = min(x + w, existing.x + existing.width)
                    y2 = min(y + h, existing.y + existing.height)

                    if x1 < x2 and y1 < y2:
                        intersection = (x2 - x1) * (y2 - y1)
                        area1 = w * h
                        area2 = existing.width * existing.height
                        iou = intersection / (area1 + area2 - intersection)

                        if iou > 0.3:
                            is_duplicate = True
                            break

                if not is_duplicate:
                    detections.append(
                        DetectedPerson(
                            x=int(x),
                            y=int(y),
                            width=int(w),
                            height=int(h),
                            confidence=0.5,
                        )
                    )

        # Expand bounding boxes to include full person shape based on color similarity
        expanded_detections = []
        for det in detections:
            exp_x, exp_y, exp_w, exp_h = self._expand_bbox_by_color_similarity(
                image, det.x, det.y, det.width, det.height, color_threshold=25
            )
            expanded_detections.append(
                DetectedPerson(
                    x=exp_x,
                    y=exp_y,
                    width=exp_w,
                    height=exp_h,
                    confidence=det.confidence,
                )
            )

        # Merge overlapping and nested bounding boxes
        merged_detections = self._merge_overlapping_boxes(
            expanded_detections, iou_threshold=0.5
        )

        return merged_detections

    def detect_from_file(self, image_path: Path) -> List[DetectedPerson]:
        """Detect people in an image file."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        return self.detect(image)

    def extract_regions(
        self,
        image: np.ndarray,
        detections: List[DetectedPerson],
        with_padding: bool = True,
    ) -> List[np.ndarray]:
        """Extract image regions for detected people."""
        regions = []
        h, w = image.shape[:2]

        for detection in detections:
            if with_padding:
                x1 = max(0, detection.x - self.config.padding)
                y1 = max(0, detection.y - self.config.padding)
                x2 = min(w, detection.x + detection.width + self.config.padding)
                y2 = min(h, detection.y + detection.height + self.config.padding)
            else:
                x1, y1 = detection.x, detection.y
                x2 = detection.x + detection.width
                y2 = detection.y + detection.height

            region = image[y1:y2, x1:x2].copy()
            regions.append(region)

        return regions

    def visualize_detections(
        self,
        image: np.ndarray,
        detections: List[DetectedPerson],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        show_confidence: bool = True,
    ) -> np.ndarray:
        """Draw bounding boxes around detected people."""
        result = image.copy()

        for detection in detections:
            x1, y1 = detection.x, detection.y
            x2 = detection.x + detection.width
            y2 = detection.y + detection.height

            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

            if show_confidence:
                label = f"{detection.confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                cv2.rectangle(
                    result,
                    (x1, y1 - label_size[1] - 4),
                    (x1 + label_size[0], y1),
                    color,
                    -1,
                )

                cv2.putText(
                    result,
                    label,
                    (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

        return result

    def save_visualized_detections(
        self,
        image_path: Path,
        output_path: Path,
        detections: Optional[List[DetectedPerson]] = None,
        **visualization_kwargs,
    ) -> Path:
        """Detect people in an image and save visualization."""
        image_path = Path(image_path)
        output_path = Path(output_path)

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        if detections is None:
            detections = self.detect(image)

        result = self.visualize_detections(image, detections, **visualization_kwargs)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result)

        return output_path


def detect_people(
    image_path: Path, config: Optional[DetectionConfig] = None
) -> List[DetectedPerson]:
    """Convenience function to detect people in an image."""
    detector = PersonDetector(config)
    return detector.detect_from_file(image_path)


def extract_competitor_regions(
    image_path: Path,
    output_dir: Path,
    config: Optional[DetectionConfig] = None,
    save_visualization: bool = True,
) -> Tuple[List[DetectedPerson], List[Path]]:
    """Detect people and extract their regions to separate files."""
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = PersonDetector(config)
    detections = detector.detect_from_file(image_path)

    if not detections:
        return detections, []

    image = cv2.imread(str(image_path))
    regions = detector.extract_regions(image, detections, with_padding=True)

    saved_paths = []
    stem = image_path.stem

    for i, region in enumerate(regions):
        region_path = output_dir / f"{stem}_person_{i:02d}.png"
        cv2.imwrite(str(region_path), region)
        saved_paths.append(region_path)

    if save_visualization:
        viz_path = output_dir / f"{stem}_detections.png"
        detector.save_visualized_detections(image_path, viz_path, detections)

    return detections, saved_paths
