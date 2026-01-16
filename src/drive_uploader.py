"""
Google Drive uploader module - handles authentication and file uploads
to Google Drive using OAuth 2.0 user credentials.
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, List, Tuple
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError


class DriveUploader:
    """Upload files to Google Drive using OAuth 2.0 user credentials"""

    SCOPES = ["https://www.googleapis.com/auth/drive.file"]

    def __init__(self, credentials_path: str, folder_id: Optional[str] = None):
        """
        Initialize Drive uploader

        Args:
            credentials_path: Path to OAuth 2.0 Client Secret JSON file
            folder_id: Google Drive folder ID where files will be uploaded
        """
        self.credentials_path = Path(credentials_path)
        self.folder_id = folder_id
        self.service = None
        self._authenticate()

    def _authenticate(self):
        """Authenticate with Google Drive API using OAuth 2.0"""
        try:
            creds = None
            # The file token.json stores the user's access and refresh tokens, and is
            # created automatically when the authorization flow completes for the first time.
            # Store token in the same directory as credentials (secrets folder)
            token_path = self.credentials_path.parent / "token.json"

            if token_path.exists():
                creds = Credentials.from_authorized_user_file(
                    str(token_path), self.SCOPES
                )

            # If there are no (valid) credentials available, let the user log in.
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if not self.credentials_path.exists():
                        raise FileNotFoundError(
                            f"Client secrets file not found at {self.credentials_path}"
                        )

                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(self.credentials_path), self.SCOPES
                    )
                    creds = flow.run_local_server(port=0)

                # Save the credentials for the next run
                with open(token_path, "w") as token:
                    token.write(creds.to_json())

            self.service = build("drive", "v3", credentials=creds)
            print("Successfully authenticated with Google Drive")
        except Exception as e:
            print(f"Error authenticating with Google Drive: {e}")
            raise

    def upload_file(
        self,
        file_path: Path,
        file_name: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> Optional[str]:
        """
        Upload a file to Google Drive

        Args:
            file_path: Path to local file
            file_name: Name for file in Drive (default: original filename)
            mime_type: MIME type (default: auto-detect)

        Returns:
            Google Drive file ID if successful, None otherwise
        """
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return None

        try:
            # Prepare file metadata
            file_metadata: Dict[str, Any] = {"name": file_name or file_path.name}

            # Add parent folder if specified
            if self.folder_id:
                file_metadata["parents"] = [self.folder_id]

            # Auto-detect MIME type if not provided
            if not mime_type:
                mime_type = self._get_mime_type(file_path)

            # Upload file
            media = MediaFileUpload(str(file_path), mimetype=mime_type, resumable=True)

            file = (
                self.service.files()
                .create(
                    body=file_metadata, media_body=media, fields="id, name, webViewLink"
                )
                .execute()
            )

            file_id = file.get("id")
            print(f"Uploaded {file_path.name} to Drive (ID: {file_id})")

            return file_id

        except HttpError as e:
            print(f"HTTP error uploading file: {e}")
            return None
        except Exception as e:
            print(f"Error uploading file: {e}")
            return None

    def upload_multiple(
        self,
        file_paths: List[Path],
        callback: Optional[Callable[[Path, Optional[str]], None]] = None,
    ) -> List[Tuple[Path, Optional[str]]]:
        """
        Upload multiple files to Google Drive

        Args:
            file_paths: List of file paths to upload
            callback: Optional callback function called after each upload (file_path, file_id)

        Returns:
            List of tuples (file_path, file_id) for each file
        """
        results: List[Tuple[Path, Optional[str]]] = []

        for file_path in file_paths:
            file_id = self.upload_file(file_path)
            results.append((file_path, file_id))

            if callback:
                callback(file_path, file_id)

        return results

    def get_folder_info(
        self, folder_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get information about a Drive folder"""
        try:
            folder_id = folder_id or self.folder_id
            if not folder_id:
                return None

            folder = (
                self.service.files()
                .get(fileId=folder_id, fields="id, name, mimeType, webViewLink")
                .execute()
            )

            return folder

        except HttpError as e:
            print(f"Error getting folder info: {e}")
            return None

    def create_folder(
        self, folder_name: str, parent_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Create a new folder in Google Drive

        Args:
            folder_name: Name for the new folder
            parent_id: Parent folder ID (optional)

        Returns:
            New folder ID if successful, None otherwise
        """
        try:
            file_metadata = {
                "name": folder_name,
                "mimeType": "application/vnd.google-apps.folder",
            }

            if parent_id:
                file_metadata["parents"] = [parent_id]

            folder = (
                self.service.files()
                .create(body=file_metadata, fields="id, name")
                .execute()
            )

            folder_id = folder.get("id")
            print(f"Created folder '{folder_name}' (ID: {folder_id})")

            return folder_id

        except HttpError as e:
            print(f"Error creating folder: {e}")
            return None

    @staticmethod
    def _get_mime_type(file_path: Path) -> str:
        """Determine MIME type from file extension"""
        ext_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }

        ext = file_path.suffix.lower()
        return ext_map.get(ext, "application/octet-stream")

    def delete_file(self, file_id: str) -> bool:
        """Delete a file from Google Drive"""
        try:
            self.service.files().delete(fileId=file_id).execute()
            print(f"Deleted file (ID: {file_id})")
            return True
        except HttpError as e:
            print(f"Error deleting file: {e}")
            return False
