"""
Image search module - handles Google Custom Search API queries
and downloads images to cache directory.
"""

import requests
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image


class ImageSearcher:
    """Search for images using Google Custom Search API"""

    def __init__(self, api_key: str, search_engine_id: str, cache_dir: str = "cache"):
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def search(
        self, query: str, num_results: int = 10, start: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Search for images using Google Custom Search API

        Args:
            query: Search query string
            num_results: Number of results to return (max 10 per request)
            start: Starting index for results (for pagination, 1-based)

        Returns:
            List of image results with url, title, dimensions, etc.
        """
        params: Dict[str, Any] = {
            "key": self.api_key,
            "cx": self.search_engine_id,
            "q": query,
            "searchType": "image",
            "num": min(num_results, 10),  # API limit is 10 per request
            "start": start,  # Starting index for pagination
            "safe": "off",  # Can be changed to "active" for safe search
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            results: List[Dict[str, Any]] = []
            if "items" in data:
                for item in data["items"]:
                    result = {
                        "url": item["link"],
                        "title": item.get("title", ""),
                        "thumbnail": item.get("image", {}).get("thumbnailLink", ""),
                        "contextLink": item.get("image", {}).get("contextLink", ""),
                        "width": item.get("image", {}).get("width"),
                        "height": item.get("image", {}).get("height"),
                        "fileFormat": item.get("fileFormat", "unknown"),
                        "mimeType": item.get("mime", ""),
                    }
                    results.append(result)

            return results

        except requests.exceptions.RequestException as e:
            print(f"Error searching images: {e}")
            return []

    def download_image(
        self, url: str, filename: Optional[str] = None, max_size_mb: float = 10
    ) -> Optional[Path]:
        """
        Download image from URL to cache directory

        Args:
            url: Image URL
            filename: Custom filename (optional, will generate from URL hash if not provided)
            max_size_mb: Maximum file size in MB

        Returns:
            Path to downloaded file or None if failed
        """
        try:
            response = requests.get(url, timeout=15, stream=True)
            response.raise_for_status()

            # Check content length
            content_length = response.headers.get("content-length")
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                if size_mb > max_size_mb:
                    print(f"Image too large: {size_mb:.2f}MB (max: {max_size_mb}MB)")
                    return None

            # Generate filename if not provided
            if not filename:
                # Extract extension from URL or content-type
                ext = self._get_extension(url, response.headers.get("content-type", ""))
                filename = f"{hash(url) % 10**10}{ext}"

            file_path = self.cache_dir / filename

            # Download and save
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Verify it's a valid image
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except Exception as e:
                print(f"Invalid image file: {e}")
                file_path.unlink()  # Delete invalid file
                return None

            return file_path

        except requests.exceptions.RequestException as e:
            print(f"Error downloading image from {url}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error downloading image: {e}")
            return None

    def download_thumbnail(
        self, thumbnail_url: str, filename: Optional[str] = None
    ) -> Optional[Path]:
        """Download thumbnail image (smaller, for preview grid)"""
        return self.download_image(thumbnail_url, filename, max_size_mb=1)

    @staticmethod
    def _get_extension(url: str, content_type: str) -> str:
        """Extract file extension from URL or content-type"""
        # Try to get from URL
        url_lower = url.lower()
        for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"]:
            if ext in url_lower:
                return ext

        # Try to get from content-type
        content_type_map = {
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "image/bmp": ".bmp",
        }

        return content_type_map.get(content_type.lower(), ".jpg")

    def get_image_info(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Get image dimensions and format"""
        try:
            with Image.open(file_path) as img:
                return {
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                    "mode": img.mode,
                    "size_bytes": file_path.stat().st_size,
                }
        except Exception as e:
            print(f"Error getting image info: {e}")
            return None
