"""
Main GUI application using PyQt6 - image search with selection and upload
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QGridLayout,
    QCheckBox,
    QMessageBox,
    QDialog,
    QProgressDialog,
    QFrame,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QMouseEvent

from image_search.tracker import ImageTracker
from image_search.searcher import ImageSearcher
from drive_manager import DriveManager


class ImagePreviewDialog(QDialog):
    """Dialog for displaying full-resolution image preview"""

    def __init__(self, image_path: Path, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Image Preview")
        self.resize(800, 600)

        layout = QVBoxLayout()

        # Create scrollable area for large images
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        # Load and display image
        label = QLabel()
        pixmap = QPixmap(str(image_path))
        label.setPixmap(pixmap)
        label.setScaledContents(False)

        scroll.setWidget(label)
        layout.addWidget(scroll)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        self.setLayout(layout)


class ImageCard(QFrame):
    """Widget representing a single image with checkbox and preview"""

    rejected = pyqtSignal(
        str
    )  # Signal emitted when reject button is clicked (url_hash)

    def __init__(
        self,
        image_data: Dict[str, Any],
        thumbnail_path: Path,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.image_data = image_data
        self.thumbnail_path = thumbnail_path
        self.full_image_path: Optional[Path] = None

        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        self.setLineWidth(1)

        layout = QVBoxLayout()

        # Thumbnail
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail_label.setFixedSize(200, 200)
        self.thumbnail_label.setScaledContents(True)
        self.thumbnail_label.mousePressEvent = lambda event: self.show_full_image(event)  # type: ignore
        self.thumbnail_label.setCursor(Qt.CursorShape.PointingHandCursor)

        pixmap = QPixmap(str(thumbnail_path))
        if not pixmap.isNull():
            self.thumbnail_label.setPixmap(
                pixmap.scaled(
                    200,
                    200,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

        layout.addWidget(self.thumbnail_label)

        # Checkbox
        self.checkbox = QCheckBox("Select for upload")
        layout.addWidget(self.checkbox)

        # Reject button
        self.reject_btn = QPushButton("Reject")
        self.reject_btn.setStyleSheet("background-color: #ffcccc;")
        self.reject_btn.clicked.connect(self.on_reject)
        layout.addWidget(self.reject_btn)

        # Image info
        info_text = f"{image_data.get('width', '?')}x{image_data.get('height', '?')}"
        info_label = QLabel(info_text)
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info_label)

        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def on_reject(self):
        """Handle reject button click"""
        if "url_hash" in self.image_data:
            self.rejected.emit(self.image_data["url_hash"])

    def show_full_image(self, event: QMouseEvent) -> None:
        """Show full resolution image in dialog"""
        if self.full_image_path and self.full_image_path.exists():
            dialog = ImagePreviewDialog(self.full_image_path, self)
            dialog.exec()
        else:
            QMessageBox.information(
                self, "Info", "Full resolution image not available yet"
            )

    def is_selected(self) -> bool:
        """Check if image is selected for upload"""
        return self.checkbox.isChecked()

    def set_full_image_path(self, path: Path):
        """Set path to full resolution image"""
        self.full_image_path = path


class SearchWorker(QThread):
    """Worker thread for searching and downloading images"""

    progress = pyqtSignal(str)  # Progress message
    result = pyqtSignal(list)  # Search results
    error = pyqtSignal(str)  # Error message
    new_start_index = pyqtSignal(int)  # Updated start index after skipping duplicates

    def __init__(
        self,
        searcher: ImageSearcher,
        tracker: ImageTracker,
        query: str,
        num_results: int,
        start_index: int = 1,
    ):
        super().__init__()
        self.searcher = searcher
        self.tracker = tracker
        self.query = query
        self.num_results = num_results
        self.start_index = start_index

    def run(self):
        """Execute search and download thumbnails"""
        try:
            self.progress.emit(f"Searching for: {self.query}")

            found_new_images = False
            current_start_index = self.start_index
            max_attempts = 5  # Prevent infinite loops (fetch up to 5 pages)
            attempts = 0

            while not found_new_images and attempts < max_attempts:
                attempts += 1
                results = self.searcher.search(
                    self.query, self.num_results, current_start_index
                )

                if not results:
                    if attempts == 1:
                        self.error.emit("No results found")
                    else:
                        self.progress.emit("No more new results found.")
                    return

                downloaded_results = []
                for i, result in enumerate(results):
                    url = result["url"]

                    # Check for duplicates
                    if self.tracker.is_duplicate(url):
                        # self.progress.emit(f"Skipping duplicate: {result.get('title', url)[:50]}")
                        continue

                    # Download thumbnail
                    self.progress.emit(f"Downloading thumbnail {i+1}/{len(results)}")
                    thumbnail_path = self.searcher.download_thumbnail(
                        result.get("thumbnail", url),
                        filename=f"thumb_{hash(url) % 10**10}.jpg",
                    )

                    if thumbnail_path:
                        result["thumbnail_path"] = thumbnail_path
                        downloaded_results.append(result)

                if downloaded_results:
                    found_new_images = True
                    self.result.emit(downloaded_results)
                    # Emit the index for the NEXT page
                    self.new_start_index.emit(current_start_index)
                else:
                    # All were duplicates, try next page
                    self.progress.emit(
                        f"Page starting at {current_start_index} contained only duplicates. Trying next page..."
                    )
                    current_start_index += self.num_results

            if not found_new_images:
                self.error.emit("No new images found (all duplicates).")

        except Exception as e:
            self.error.emit(f"Error during search: {str(e)}")


class UploadWorker(QThread):
    """Worker thread for uploading images to Drive"""

    progress = pyqtSignal(str, int, int)  # message, current, total
    finished = pyqtSignal(int, int)  # uploaded_count, failed_count
    error = pyqtSignal(str)

    def __init__(
        self,
        uploader: DriveManager,
        tracker: ImageTracker,
        images: List[Tuple[str, Path]],
    ):
        super().__init__()
        self.uploader = uploader
        self.tracker = tracker
        self.images = images  # List of (url_hash, file_path)

    def run(self):
        """Upload selected images"""
        try:
            uploaded = 0
            failed = 0
            total = len(self.images)

            for i, (url_hash, file_path) in enumerate(self.images):
                # Upload the original file
                self.progress.emit(f"Uploading {file_path.name}", i + 1, total)

                file_id = self.uploader.upload_file(file_path)

                if file_id:
                    self.tracker.update_status(url_hash, "uploaded", file_id)
                    uploaded += 1
                else:
                    self.tracker.update_status(url_hash, "failed")
                    failed += 1

            self.finished.emit(uploaded, failed)

        except Exception as e:
            self.error.emit(f"Error during upload: {str(e)}")


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self, config_path: str = "secrets/config.json"):
        super().__init__()

        # Load configuration
        self.config = self.load_config(config_path)

        # Initialize components
        self.tracker = ImageTracker(self.config["tracking"]["file"])
        self.searcher = ImageSearcher(
            self.config["google_custom_search"]["api_key"],
            self.config["google_custom_search"]["search_engine_id"],
            self.config["cache"]["directory"],
        )

        try:
            self.uploader = DriveManager(
                self.config["google_drive"]["credentials_path"],
                self.config["google_drive"]["folder_id"],
            )
        except Exception as e:
            QMessageBox.warning(
                self,
                "Drive Setup",
                f"Could not initialize Google Drive: {e}\nUpload functionality will be disabled.",
            )
            self.uploader = None

        # UI state
        self.image_cards: List[ImageCard] = []
        self.search_worker = None
        self.upload_worker = None
        self.current_search_query = ""
        self.current_start_index = 1

        self.init_ui()

        # Load images from previous session
        self.load_previous_session()

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        with open(config_path, "r") as f:
            return json.load(f)

    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("Image Search & Upload")
        self.resize(1200, 800)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Search section
        search_layout = QHBoxLayout()

        search_label = QLabel("Search Query:")
        search_layout.addWidget(search_label)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText(
            "Enter search terms (e.g., 'mountain landscape')"
        )
        self.search_input.returnPressed.connect(self.start_search)
        search_layout.addWidget(self.search_input)

        self.search_btn = QPushButton("Search")
        self.search_btn.clicked.connect(self.start_search)
        search_layout.addWidget(self.search_btn)

        main_layout.addLayout(search_layout)

        # Status label
        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)

        # Image grid (scrollable)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout()
        self.grid_widget.setLayout(self.grid_layout)
        scroll_area.setWidget(self.grid_widget)

        main_layout.addWidget(scroll_area)

        # Action buttons
        action_layout = QHBoxLayout()

        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.select_all)
        action_layout.addWidget(self.select_all_btn)

        self.deselect_all_btn = QPushButton("Deselect All")
        self.deselect_all_btn.clicked.connect(self.deselect_all)
        action_layout.addWidget(self.deselect_all_btn)

        self.load_more_btn = QPushButton("Load More Results")
        self.load_more_btn.clicked.connect(self.load_more_results)
        self.load_more_btn.setEnabled(False)
        action_layout.addWidget(self.load_more_btn)

        action_layout.addStretch()

        self.upload_btn = QPushButton("Upload Selected to Drive (auto preprocess)")
        self.upload_btn.clicked.connect(self.upload_selected)
        self.upload_btn.setEnabled(False)
        action_layout.addWidget(self.upload_btn)

        main_layout.addLayout(action_layout)

    def start_search(self):
        """Start image search"""
        query = self.search_input.text().strip()
        if not query:
            QMessageBox.warning(self, "Input Error", "Please enter a search query")
            return

        # Clear previous results and reset pagination
        self.clear_grid()
        self.current_search_query = query
        self.current_start_index = 1

        # Disable buttons
        self.search_btn.setEnabled(False)
        self.upload_btn.setEnabled(False)
        self.load_more_btn.setEnabled(False)

        # Start search worker
        num_results = self.config["google_custom_search"]["num_results"]
        self.search_worker = SearchWorker(
            self.searcher, self.tracker, query, num_results, self.current_start_index
        )
        self.search_worker.progress.connect(self.update_status)
        self.search_worker.result.connect(self.display_results)
        self.search_worker.new_start_index.connect(self.update_start_index)
        self.search_worker.error.connect(self.show_error)
        self.search_worker.finished.connect(lambda: self.search_btn.setEnabled(True))
        self.search_worker.start()

    def load_more_results(self):
        """Load more results for the current search query"""
        if not self.current_search_query:
            return

        # Increment start index for next page
        self.current_start_index += self.config["google_custom_search"]["num_results"]

        # Disable buttons during search
        self.search_btn.setEnabled(False)
        self.load_more_btn.setEnabled(False)

        # Start search worker with new start index
        num_results = self.config["google_custom_search"]["num_results"]
        self.search_worker = SearchWorker(
            self.searcher,
            self.tracker,
            self.current_search_query,
            num_results,
            self.current_start_index,
        )
        self.search_worker.progress.connect(self.update_status)
        self.search_worker.result.connect(self.display_results)
        self.search_worker.new_start_index.connect(self.update_start_index)
        self.search_worker.error.connect(self.show_error)
        self.search_worker.finished.connect(self.on_load_more_finished)
        self.search_worker.start()

    def update_start_index(self, index: int):
        """Update current start index from worker"""
        self.current_start_index = index

    def on_load_more_finished(self):
        """Handle load more completion"""
        self.search_btn.setEnabled(True)
        self.load_more_btn.setEnabled(True)

    def display_results(self, results: List[Dict[str, Any]]) -> None:
        """Display search results in grid"""
        self.update_status(f"Found {len(results)} images")

        if not results:
            self.load_more_btn.setEnabled(False)
            return

        # Create image cards
        columns = 4
        existing_cards = len(self.image_cards)
        for i, result in enumerate(results):
            card = ImageCard(result, result["thumbnail_path"])
            card.rejected.connect(self.reject_image)  # Connect reject signal
            self.image_cards.append(card)

            card_index = existing_cards + i
            row = card_index // columns
            col = card_index % columns
            self.grid_layout.addWidget(card, row, col)

            # Download full resolution image in background
            self.download_full_image(result, card)

        self.upload_btn.setEnabled(True)
        self.load_more_btn.setEnabled(True)

    def reject_image(self, url_hash: str):
        """Handle image rejection"""
        # Update tracker status
        self.tracker.update_status(url_hash, "rejected")

        # Find and remove card
        card_to_remove = None
        for card in self.image_cards:
            if card.image_data.get("url_hash") == url_hash:
                card_to_remove = card
                break

        if card_to_remove:
            self.image_cards.remove(card_to_remove)
            self.grid_layout.removeWidget(card_to_remove)
            card_to_remove.deleteLater()

            # Re-layout remaining cards
            columns = 4
            for i, card in enumerate(self.image_cards):
                self.grid_layout.removeWidget(card)
                row = i // columns
                col = i % columns
                self.grid_layout.addWidget(card, row, col)

            self.update_status("Image rejected")

    def download_full_image(self, result: Dict[str, Any], card: ImageCard) -> None:
        """Download full resolution image for a card"""
        url = result["url"]
        file_path = self.searcher.download_image(url)

        if file_path:
            # Check for content duplicate (file hash)
            existing_image = self.tracker.get_image_by_file_hash(file_path)
            if existing_image:
                print(
                    f"Duplicate content detected for {url}. Matches {existing_image['url']}"
                )
                self.remove_card(card)
                try:
                    file_path.unlink()
                except Exception as e:
                    print(f"Error deleting duplicate file: {e}")
                return

            card.set_full_image_path(file_path)

            # Convert Path objects to strings in metadata
            metadata = {
                k: str(v) if isinstance(v, Path) else v for k, v in result.items()
            }

            # Add to tracker
            url_hash = self.tracker.add_image(
                url=url,
                search_query=self.search_input.text(),
                file_path=file_path,
                status="pending",
                metadata=metadata,
            )
            card.image_data["url_hash"] = url_hash
            card.image_data["file_path"] = file_path
        else:
            # Download failed - remove from UI and track as failed
            print(f"Failed to download full image: {url}")

            # Convert Path objects to strings in metadata
            metadata = {
                k: str(v) if isinstance(v, Path) else v for k, v in result.items()
            }

            # Add to tracker as failed so we know about it
            self.tracker.add_image(
                url=url,
                search_query=self.search_input.text(),
                file_path=None,
                status="failed",
                metadata=metadata,
            )

            self.remove_card(card)

    def remove_card(self, card: ImageCard):
        """Remove card from grid and re-layout"""
        if card in self.image_cards:
            self.image_cards.remove(card)
            self.grid_layout.removeWidget(card)
            card.deleteLater()

            # Re-layout remaining cards
            columns = 4
            for i, c in enumerate(self.image_cards):
                self.grid_layout.removeWidget(c)
                row = i // columns
                col = i % columns
                self.grid_layout.addWidget(c, row, col)

    def upload_selected(self):
        """Upload selected images to Google Drive"""
        if not self.uploader:
            QMessageBox.warning(
                self, "Upload Error", "Google Drive uploader not initialized"
            )
            return

        # Get selected images
        selected = []
        for card in self.image_cards:
            if (
                card.is_selected()
                and "url_hash" in card.image_data
                and "file_path" in card.image_data
            ):
                selected.append(
                    (card.image_data["url_hash"], card.image_data["file_path"])
                )

        if not selected:
            QMessageBox.information(
                self, "No Selection", "Please select at least one image to upload"
            )
            return

        # Update selected images status
        for url_hash, _ in selected:
            self.tracker.update_status(url_hash, "selected")

        # Disable buttons
        self.upload_btn.setEnabled(False)
        self.search_btn.setEnabled(False)

        # Start upload worker
        self.upload_worker = UploadWorker(
            self.uploader,
            self.tracker,
            selected,
        )
        self.upload_worker.progress.connect(self.update_upload_progress)
        self.upload_worker.finished.connect(self.upload_finished)
        self.upload_worker.error.connect(self.show_error)
        self.upload_worker.start()

        # Show progress dialog
        self.progress_dialog = QProgressDialog(
            "Uploading images...", "Cancel", 0, len(selected), self
        )
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.show()

    def update_upload_progress(self, message: str, current: int, total: int):
        """Update upload progress"""
        self.update_status(message)
        if hasattr(self, "progress_dialog"):
            self.progress_dialog.setValue(current)
            self.progress_dialog.setLabelText(message)

    def upload_finished(self, uploaded: int, failed: int):
        """Handle upload completion"""
        if hasattr(self, "progress_dialog"):
            self.progress_dialog.close()

        self.search_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)

        QMessageBox.information(
            self,
            "Upload Complete",
            f"Successfully uploaded: {uploaded}\nFailed: {failed}",
        )

        self.update_status(f"Upload complete: {uploaded} successful, {failed} failed")

    def clear_grid(self):
        """Clear image grid"""
        for card in self.image_cards:
            card.deleteLater()
        self.image_cards.clear()

    def load_previous_session(self):
        """Load images from previous session that haven't been uploaded yet"""
        # Get images with status pending, selected, or failed (not uploaded or rejected)
        pending_images = []
        for img in self.tracker.data["images"]:
            if img["status"] in ["pending", "selected", "failed"]:
                pending_images.append(img)

        if not pending_images:
            return

        # Display loaded images
        self.update_status(
            f"Loading {len(pending_images)} images from previous session"
        )

        columns = 4
        loaded_count = 0

        for img_data in pending_images:
            # Check if files still exist
            raw_local_path = img_data.get("localPath")
            local_path = Path(raw_local_path) if raw_local_path else None

            # Skip if full resolution image is missing (as per user request)
            if not local_path or not local_path.exists():
                continue

            raw_thumb_path = img_data.get("metadata", {}).get("thumbnail_path")
            thumbnail_path = Path(raw_thumb_path) if raw_thumb_path else None

            # Use existing thumbnail or full image for display
            display_path = (
                thumbnail_path
                if (thumbnail_path and thumbnail_path.exists())
                else local_path
            )

            # Create a result dict compatible with ImageCard
            result = {
                "url": img_data["url"],
                "thumbnail_path": display_path,
                "width": img_data.get("metadata", {}).get("width", "?"),
                "height": img_data.get("metadata", {}).get("height", "?"),
            }

            # Create card
            card = ImageCard(result, display_path)
            card.set_full_image_path(local_path)
            card.rejected.connect(self.reject_image)  # Connect reject signal

            # Store tracking info
            card.image_data["url_hash"] = img_data["urlHash"]
            card.image_data["file_path"] = local_path

            # Pre-select if it was selected before
            if img_data["status"] == "selected":
                card.checkbox.setChecked(True)

            self.image_cards.append(card)

            row = loaded_count // columns
            col = loaded_count % columns
            self.grid_layout.addWidget(card, row, col)

            loaded_count += 1

        if loaded_count > 0:
            self.upload_btn.setEnabled(True)
            self.update_status(f"Loaded {loaded_count} images from previous session")
        else:
            self.update_status("Ready")

    def select_all(self):
        """Select all images"""
        for card in self.image_cards:
            card.checkbox.setChecked(True)

    def deselect_all(self):
        """Deselect all images"""
        for card in self.image_cards:
            card.checkbox.setChecked(False)

    def update_status(self, message: str):
        """Update status label"""
        self.status_label.setText(message)

    def show_error(self, error_msg: str):
        """Show error message"""
        QMessageBox.critical(self, "Error", error_msg)
        self.update_status(f"Error: {error_msg}")


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
