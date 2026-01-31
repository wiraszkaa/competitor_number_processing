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
class NumberRegionCandidate:
    """Reprezentuje kandydata na region z numerem."""
    
    # Współrzędne bounding box (x, y, width, height)
    x: int
    y: int
    width: int
    height: int
    
    # Wynik pewności (0.0 - 1.0)
    confidence: float
    
    # Metoda detekcji użyta do znalezienia regionu
    method: str  # 'canny', 'hsv', 'adaptive', 'mser', 'combined'
    
    # Dodatkowe cechy regionu
    aspect_ratio: float
    fill_ratio: float  # Gęstość wypełnienia
    edge_density: float  # Gęstość krawędzi
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Zwraca bounding box jako (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)
    
    @property
    def center(self) -> Tuple[int, int]:
        """Zwraca środek regionu."""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def area(self) -> int:
        """Zwraca powierzchnię regionu."""
        return self.width * self.height


@dataclass(frozen=True)
class NumberDetectionConfig:
    """Konfiguracja dla detekcji numerów."""
    
    # Minimalne/maksymalne wymiary regionu z numerem (szerokość, wysokość)
    min_region_size: Tuple[int, int] = (20, 35)  # Większe minimum dla lepszej jakości
    max_region_size: Tuple[int, int] = (150, 250)  # Mniejsze max - numery nie są tak duże
    
    # Zakres proporcji aspect ratio (wysokość/szerokość) - bardziej restrykcyjne dla cyfr
    min_aspect_ratio: float = 1.0  # Cyfry są wyższe niż szersze
    max_aspect_ratio: float = 3.5  # Nie za wąskie
    
    # Minimalna gęstość wypełnienia (fill ratio)
    min_fill_ratio: float = 0.25  # Wyższe minimum - cyfry mają więcej wypełnienia
    max_fill_ratio: float = 0.85  # Niższe max - nie całe pełne bloki
    
    # Minimalna gęstość krawędzi
    min_edge_density: float = 0.08  # Wyższe minimum - cyfry mają krawędzie
    
    # Parametry dla Canny (nieużywane gdy używamy tylko MSER)
    canny_low: int = 50
    canny_high: int = 150
    
    # Parametry dla adaptive threshold (nieużywane)
    adaptive_block_size: int = 11
    adaptive_c: int = 2
    
    # Czy używać MSER - TYLKO MSER jest włączony
    use_mser: bool = True
    use_canny: bool = False  # Wyłączone
    use_hsv: bool = False     # Wyłączone
    use_adaptive: bool = False  # Wyłączone
    
    # Próg IoU dla non-maximum suppression
    nms_iou_threshold: float = 0.4  # Bardziej agresywne NMS
    
    # Minimalna pewność do zaakceptowania kandydata - wyższe dla lepszej jakości
    min_confidence: float = 0.5
    
    # Maksymalna liczba kandydatów na osobę
    max_candidates_per_person: int = 3  # Zmniejszone do 3 (koszulka + ewentualnie spodenki)
    
    # Czy grupować cyfry w numery 2-cyfrowe (domyślnie wyłączone bo numery mogą być 1-cyfrowe)
    group_digits: bool = False
    
    # Maksymalna odległość między cyframi w numerze (w pikselach) - używane gdy group_digits=True
    max_digit_distance: int = 50


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


class NumberRegionDetector:
    """
    Detektor obszarów kandydatów na numery graczy.
    
    Implementuje różne metody detekcji:
    1. Detekcja krawędzi (Canny) i konturów
    2. Segmentacja po kolorze (HSV thresholding)
    3. Adaptive threshold + connected components
    4. MSER dla detekcji regionów tekstowych
    5. Filtrowanie po rozmiarze, proporcjach i gęstości
    """
    
    def __init__(self, config: Optional[NumberDetectionConfig] = None):
        """Inicjalizacja detektora regionów z numerami."""
        self.config = config or NumberDetectionConfig()
        
        # Inicjalizuj MSER jeśli włączony (TYLKO MSER jest używany)
        if self.config.use_mser:
            # MSER parametry bez prefiksów "_" - bardziej restrykcyjne dla cyfr
            self.mser = cv2.MSER_create()
            # Ustaw parametry po utworzeniu
            self.mser.setDelta(8)  # Większa delta dla stabilniejszych regionów
            self.mser.setMinArea(80)  # Większe minimum - pomija drobny szum
            self.mser.setMaxArea(10000)  # Mniejsze max - cyfry nie są ogromne
            self.mser.setMaxVariation(0.3)  # Mniejsza wariancja dla stabilnych regionów
            self.mser.setMinDiversity(0.3)  # Większa różnorodność
    
    def detect_candidates(
        self, 
        region_image: np.ndarray,
        use_methods: Optional[List[str]] = None
    ) -> List[NumberRegionCandidate]:
        """
        Wykrywa kandydatów na obszary z numerami używając wielu metod.
        
        Args:
            region_image: Obraz regionu z wykrytą osobą (BGR)
            use_methods: Lista metod do użycia. Jeśli None, używa wszystkich.
                        Dostępne: ['canny', 'hsv', 'adaptive', 'mser']
        
        Returns:
            Lista kandydatów na regiony z numerami
        """
        # Automatycznie wybierz metody na podstawie konfiguracji
        if use_methods is None:
            use_methods = []
            if self.config.use_canny:
                use_methods.append('canny')
            if self.config.use_hsv:
                use_methods.append('hsv')
            if self.config.use_adaptive:
                use_methods.append('adaptive')
            if self.config.use_mser:
                use_methods.append('mser')
        
        all_candidates = []
        
        # 1. Detekcja krawędzi (Canny) i konturów - WYŁĄCZONE
        if 'canny' in use_methods and self.config.use_canny:
            candidates = self._detect_by_canny(region_image)
            all_candidates.extend(candidates)
        
        # 2. Segmentacja po kolorze (HSV thresholding) - WYŁĄCZONE
        if 'hsv' in use_methods and self.config.use_hsv:
            candidates = self._detect_by_hsv(region_image)
            all_candidates.extend(candidates)
        
        # 3. Adaptive threshold + connected components - WYŁĄCZONE
        if 'adaptive' in use_methods and self.config.use_adaptive:
            candidates = self._detect_by_adaptive(region_image)
            all_candidates.extend(candidates)
        
        # 4. MSER dla detekcji regionów tekstowych - TYLKO TA METODA JEST WŁĄCZONA
        if 'mser' in use_methods and self.config.use_mser:
            candidates = self._detect_by_mser(region_image)
            all_candidates.extend(candidates)
        
        # 5. Filtrowanie i non-maximum suppression
        filtered_candidates = self._filter_candidates(all_candidates)
        nms_candidates = self._non_max_suppression(filtered_candidates)
        
        # 6. Ograniczenie do max 3 kandydatów (dla lepszej jakości)
        final_candidates = self._limit_candidates_per_person(nms_candidates)
        
        return final_candidates
    
    def _detect_by_canny(self, image: np.ndarray) -> List[NumberRegionCandidate]:
        """
        Detekcja krawędzi metodą Canny i ekstrakcja konturów.
        
        Args:
            image: Obraz wejściowy (BGR)
            
        Returns:
            Lista kandydatów wykrytych metodą Canny
        """
        # Konwersja do skali szarości
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Equalizacja histogramu dla lepszego kontrastu
        gray = cv2.equalizeHist(gray)
        
        # Detekcja krawędzi Canny
        edges = cv2.Canny(
            gray, 
            self.config.canny_low, 
            self.config.canny_high
        )
        
        # Dylatacja krawędzi aby połączyć blisko siebie
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Znajdź kontury
        contours, _ = cv2.findContours(
            edges, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        candidates = []
        for contour in contours:
            candidate = self._contour_to_candidate(
                contour, 
                image, 
                edges, 
                method='canny'
            )
            if candidate:
                candidates.append(candidate)
        
        return candidates
    
    def _detect_by_hsv(self, image: np.ndarray) -> List[NumberRegionCandidate]:
        """
        Segmentacja po kolorze w przestrzeni HSV.
        Szuka regionów o jasnych kolorach (białe/żółte numery na koszulkach).
        
        Args:
            image: Obraz wejściowy (BGR)
            
        Returns:
            Lista kandydatów wykrytych metodą HSV
        """
        # Konwersja do HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Dwie maski: białe i żółte/jasne kolory
        # Maska 1: Białe/bardzo jasne (niskie nasycenie, wysoka wartość)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 60, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        # Maska 2: Żółte/pomarańczowe (typowe kolory numerów)
        lower_yellow = np.array([15, 100, 150])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Połącz maski
        mask = cv2.bitwise_or(mask_white, mask_yellow)
        
        # Morfologia: usuń szum i połącz bliskie regiony
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        
        # Znajdź kontury
        contours, _ = cv2.findContours(
            mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Oblicz gęstość krawędzi dla każdego regionu
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        candidates = []
        for contour in contours:
            candidate = self._contour_to_candidate(
                contour, 
                image, 
                edges, 
                method='hsv'
            )
            if candidate:
                candidates.append(candidate)
        
        return candidates
    
    def _detect_by_adaptive(self, image: np.ndarray) -> List[NumberRegionCandidate]:
        """
        Adaptive thresholding + connected components.
        Dobra dla tekstu o różnym oświetleniu.
        
        Args:
            image: Obraz wejściowy (BGR)
            
        Returns:
            Lista kandydatów wykrytych metodą adaptive threshold
        """
        # Konwersja do skali szarości
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Denoising
        gray = cv2.fastNlMeansDenoising(gray, None, h=10)
        
        # Adaptive threshold - dwie wersje (dla jasnego i ciemnego tekstu)
        binary1 = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.config.adaptive_block_size,
            self.config.adaptive_c
        )
        
        binary2 = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.config.adaptive_block_size,
            self.config.adaptive_c
        )
        
        # Połącz obie maski
        binary = cv2.bitwise_or(binary1, binary2)
        
        # Morfologia: usuń szum
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, 
            connectivity=8
        )
        
        # Oblicz krawędzie dla walidacji
        edges = cv2.Canny(gray, 50, 150)
        
        candidates = []
        # Pomiń tło (label 0)
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Podstawowa walidacja rozmiaru
            if w < self.config.min_region_size[0] or h < self.config.min_region_size[1]:
                continue
            if w > self.config.max_region_size[0] or h > self.config.max_region_size[1]:
                continue
            
            # Oblicz cechy
            aspect_ratio = h / w if w > 0 else 0
            fill_ratio = area / (w * h) if (w * h) > 0 else 0
            
            # Gęstość krawędzi
            roi_edges = edges[y:y+h, x:x+w]
            edge_density = np.sum(roi_edges > 0) / (w * h) if (w * h) > 0 else 0
            
            # Pewność bazowana na cechach
            confidence = self._calculate_confidence(
                aspect_ratio, 
                fill_ratio, 
                edge_density
            )
            
            candidate = NumberRegionCandidate(
                x=x, y=y, width=w, height=h,
                confidence=confidence,
                method='adaptive',
                aspect_ratio=aspect_ratio,
                fill_ratio=fill_ratio,
                edge_density=edge_density
            )
            candidates.append(candidate)
        
        return candidates
    
    def _detect_by_mser(self, image: np.ndarray) -> List[NumberRegionCandidate]:
        """
        MSER (Maximally Stable Extremal Regions) dla detekcji tekstu.
        Bardzo dobra dla wykrywania cyfr i liter.
        
        Args:
            image: Obraz wejściowy (BGR)
            
        Returns:
            Lista kandydatów wykrytych metodą MSER
        """
        if not self.config.use_mser:
            return []
        
        # Konwersja do skali szarości
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Wykryj regiony MSER
        regions, bboxes = self.mser.detectRegions(gray)
        
        # Oblicz krawędzie dla walidacji
        edges = cv2.Canny(gray, 50, 150)
        
        candidates = []
        for bbox in bboxes:
            x, y, w, h = bbox
            
            # Podstawowa walidacja rozmiaru
            if w < self.config.min_region_size[0] or h < self.config.min_region_size[1]:
                continue
            if w > self.config.max_region_size[0] or h > self.config.max_region_size[1]:
                continue
            
            # Oblicz cechy
            aspect_ratio = h / w if w > 0 else 0
            
            # Fill ratio dla MSER można oszacować z regionu
            roi = gray[y:y+h, x:x+w]
            _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            fill_ratio = np.sum(binary > 0) / (w * h) if (w * h) > 0 else 0
            
            # Gęstość krawędzi
            roi_edges = edges[y:y+h, x:x+w]
            edge_density = np.sum(roi_edges > 0) / (w * h) if (w * h) > 0 else 0
            
            # Pewność
            confidence = self._calculate_confidence(
                aspect_ratio, 
                fill_ratio, 
                edge_density
            )
            
            candidate = NumberRegionCandidate(
                x=x, y=y, width=w, height=h,
                confidence=confidence,
                method='mser',
                aspect_ratio=aspect_ratio,
                fill_ratio=fill_ratio,
                edge_density=edge_density
            )
            candidates.append(candidate)
        
        return candidates
    
    def _contour_to_candidate(
        self, 
        contour: np.ndarray,
        image: np.ndarray,
        edges: np.ndarray,
        method: str
    ) -> Optional[NumberRegionCandidate]:
        """
        Konwertuje kontur na kandydata z walidacją.
        
        Args:
            contour: Kontur OpenCV
            image: Oryginalny obraz
            edges: Obraz krawędzi
            method: Nazwa metody detekcji
            
        Returns:
            Kandydat lub None jeśli nie spełnia kryteriów
        """
        # Oblicz bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Podstawowa walidacja rozmiaru
        if w < self.config.min_region_size[0] or h < self.config.min_region_size[1]:
            return None
        if w > self.config.max_region_size[0] or h > self.config.max_region_size[1]:
            return None
        
        # Oblicz cechy
        aspect_ratio = h / w if w > 0 else 0
        
        # Fill ratio - stosunek powierzchni konturu do bbox
        contour_area = cv2.contourArea(contour)
        fill_ratio = contour_area / (w * h) if (w * h) > 0 else 0
        
        # Gęstość krawędzi w regionie
        roi_edges = edges[y:y+h, x:x+w]
        edge_density = np.sum(roi_edges > 0) / (w * h) if (w * h) > 0 else 0
        
        # Oblicz pewność
        confidence = self._calculate_confidence(
            aspect_ratio, 
            fill_ratio, 
            edge_density
        )
        
        return NumberRegionCandidate(
            x=x, y=y, width=w, height=h,
            confidence=confidence,
            method=method,
            aspect_ratio=aspect_ratio,
            fill_ratio=fill_ratio,
            edge_density=edge_density
        )
    
    def _calculate_confidence(
        self, 
        aspect_ratio: float, 
        fill_ratio: float, 
        edge_density: float
    ) -> float:
        """
        Oblicza pewność kandydata bazując na jego cechach.
        
        Args:
            aspect_ratio: Proporcje (wysokość/szerokość)
            fill_ratio: Gęstość wypełnienia
            edge_density: Gęstość krawędzi
            
        Returns:
            Pewność w zakresie [0, 1]
        """
        score = 0.0
        
        # Punkty za dobry aspect ratio
        if self.config.min_aspect_ratio <= aspect_ratio <= self.config.max_aspect_ratio:
            # Im bliżej optymalnego (1.5-2.5), tym lepiej
            if 1.5 <= aspect_ratio <= 2.5:
                score += 0.4
            else:
                score += 0.2
        
        # Punkty za dobry fill ratio
        if self.config.min_fill_ratio <= fill_ratio <= self.config.max_fill_ratio:
            # Optymalne fill ratio dla cyfr: 0.3-0.6
            if 0.3 <= fill_ratio <= 0.6:
                score += 0.3
            else:
                score += 0.15
        
        # Punkty za dobrą gęstość krawędzi
        if edge_density >= self.config.min_edge_density:
            # Optymalna gęstość: 0.1-0.3
            if 0.1 <= edge_density <= 0.3:
                score += 0.3
            else:
                score += 0.15
        
        return min(1.0, score)
    
    def _filter_candidates(
        self, 
        candidates: List[NumberRegionCandidate]
    ) -> List[NumberRegionCandidate]:
        """
        Filtruje kandydatów po rozmiarze, proporcjach i gęstości.
        
        Args:
            candidates: Lista wszystkich kandydatów
            
        Returns:
            Lista przefiltrowanych kandydatów
        """
        filtered = []
        
        for candidate in candidates:
            # Sprawdź aspect ratio
            if not (self.config.min_aspect_ratio <= candidate.aspect_ratio <= self.config.max_aspect_ratio):
                continue
            
            # Sprawdź fill ratio
            if not (self.config.min_fill_ratio <= candidate.fill_ratio <= self.config.max_fill_ratio):
                continue
            
            # Sprawdź edge density
            if candidate.edge_density < self.config.min_edge_density:
                continue
            
            # Sprawdź minimalną pewność
            if candidate.confidence < self.config.min_confidence:
                continue
            
            filtered.append(candidate)
        
        return filtered
    
    def _non_max_suppression(
        self, 
        candidates: List[NumberRegionCandidate]
    ) -> List[NumberRegionCandidate]:
        """
        Non-maximum suppression - usuwa nakładające się kandydatów.
        
        Args:
            candidates: Lista kandydatów
            
        Returns:
            Lista kandydatów po NMS
        """
        if len(candidates) == 0:
            return []
        
        # Sortuj po pewności (malejąco)
        sorted_candidates = sorted(
            candidates, 
            key=lambda c: c.confidence, 
            reverse=True
        )
        
        keep = []
        
        while sorted_candidates:
            # Weź kandydata o najwyższej pewności
            best = sorted_candidates.pop(0)
            keep.append(best)
            
            # Usuń kandydatów zbyt podobnych
            remaining = []
            for candidate in sorted_candidates:
                iou = self._calculate_iou(best, candidate)
                if iou < self.config.nms_iou_threshold:
                    remaining.append(candidate)
            
            sorted_candidates = remaining
        
        return keep
    
    def _calculate_iou(
        self, 
        candidate1: NumberRegionCandidate, 
        candidate2: NumberRegionCandidate
    ) -> float:
        """
        Oblicza Intersection over Union między dwoma kandydatami.
        
        Args:
            candidate1: Pierwszy kandydat
            candidate2: Drugi kandydat
            
        Returns:
            IoU w zakresie [0, 1]
        """
        # Współrzędne przecięcia
        x1 = max(candidate1.x, candidate2.x)
        y1 = max(candidate1.y, candidate2.y)
        x2 = min(candidate1.x + candidate1.width, candidate2.x + candidate2.width)
        y2 = min(candidate1.y + candidate1.height, candidate2.y + candidate2.height)
        
        # Powierzchnia przecięcia
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Powierzchnie
        area1 = candidate1.area
        area2 = candidate2.area
        
        # Union
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _limit_candidates_per_person(
        self, 
        candidates: List[NumberRegionCandidate]
    ) -> List[NumberRegionCandidate]:
        """
        Ogranicza liczbę kandydatów do max 4 na osobę.
        Numery graczy są 2-cyfrowe i mogą być na koszulce i spodenkach.
        Max: 2 numery × 2 cyfry = 4 kandydaty.
        
        Args:
            candidates: Lista wszystkich kandydatów
            
        Returns:
            Lista max 4 najlepszych kandydatów
        """
        if len(candidates) <= self.config.max_candidates_per_person:
            return candidates
        
        # Sortuj po pewności (malejąco) i weź top N
        sorted_candidates = sorted(
            candidates, 
            key=lambda c: c.confidence, 
            reverse=True
        )
        
        return sorted_candidates[:self.config.max_candidates_per_person]
    
    def group_into_numbers(
        self, 
        candidates: List[NumberRegionCandidate]
    ) -> List[Tuple[NumberRegionCandidate, Optional[NumberRegionCandidate]]]:
        """
        Grupuje cyfry w numery 2-cyfrowe.
        Znajduje pary cyfr obok siebie (poziomo lub pionowo).
        
        Args:
            candidates: Lista kandydatów (cyfr)
            
        Returns:
            Lista par (cyfra1, cyfra2) gdzie cyfra2 może być None dla pojedynczych cyfr
        """
        if not self.config.group_digits:
            return [(c, None) for c in candidates]
        
        if len(candidates) == 0:
            return []
        
        # Sortuj kandydatów po pozycji (lewa->prawa, góra->dół)
        sorted_candidates = sorted(
            candidates,
            key=lambda c: (c.y, c.x)
        )
        
        used = set()
        pairs = []
        
        for i, candidate1 in enumerate(sorted_candidates):
            if i in used:
                continue
            
            best_match = None
            best_distance = float('inf')
            best_idx = None
            
            # Szukaj najbliższego kandydata
            for j, candidate2 in enumerate(sorted_candidates):
                if i == j or j in used:
                    continue
                
                # Oblicz odległość między środkami
                cx1, cy1 = candidate1.center
                cx2, cy2 = candidate2.center
                distance = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
                
                # Sprawdź czy są w tej samej linii (poziomo lub pionowo)
                vertical_diff = abs(cy2 - cy1)
                horizontal_diff = abs(cx2 - cx1)
                
                # Numery są zwykle obok siebie poziomo
                is_horizontal = horizontal_diff > vertical_diff
                
                # Sprawdź czy są wystarczająco blisko
                if distance <= self.config.max_digit_distance:
                    # Preferuj pary poziome (typowe dla numerów)
                    if is_horizontal:
                        distance *= 0.8  # Bonus dla par poziomych
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_match = candidate2
                        best_idx = j
            
            if best_match is not None:
                # Znaleziono parę
                # Upewnij się że lewa cyfra jest pierwsza
                if candidate1.x <= best_match.x:
                    pairs.append((candidate1, best_match))
                else:
                    pairs.append((best_match, candidate1))
                used.add(i)
                used.add(best_idx)
            else:
                # Pojedyncza cyfra (może być numer 1-cyfrowy lub niepełny)
                pairs.append((candidate1, None))
                used.add(i)
        
        return pairs
    
    def visualize_candidates(
        self,
        image: np.ndarray,
        candidates: List[NumberRegionCandidate],
        show_method: bool = True,
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        Wizualizuje wykrytych kandydatów na obrazie.
        
        Args:
            image: Obraz wejściowy
            candidates: Lista kandydatów
            show_method: Czy pokazywać metodę detekcji
            show_confidence: Czy pokazywać pewność
            
        Returns:
            Obraz z wizualizacją
        """
        result = image.copy()
        
        # Kolory dla różnych metod
        method_colors = {
            'canny': (0, 255, 0),      # Zielony
            'hsv': (255, 0, 0),         # Niebieski
            'adaptive': (0, 165, 255),  # Pomarańczowy
            'mser': (255, 0, 255),      # Magenta
            'combined': (0, 255, 255)   # Żółty
        }
        
        for candidate in candidates:
            # Wybierz kolor
            color = method_colors.get(candidate.method, (255, 255, 255))
            
            # Rysuj bounding box
            x, y, w, h = candidate.bbox
            thickness = 2 if candidate.confidence > 0.6 else 1
            cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
            
            # Etykieta
            label_parts = []
            if show_method:
                label_parts.append(candidate.method[:3])
            if show_confidence:
                label_parts.append(f"{candidate.confidence:.2f}")
            
            if label_parts:
                label = " ".join(label_parts)
                
                # Tło dla tekstu
                (label_w, label_h), _ = cv2.getTextSize(
                    label, 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4, 
                    1
                )
                cv2.rectangle(
                    result,
                    (x, y - label_h - 4),
                    (x + label_w, y),
                    color,
                    -1
                )
                
                # Tekst
                cv2.putText(
                    result,
                    label,
                    (x, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 0),
                    1
                )
        
        return result
