from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import os

import conf


# ---- KONFIG ----
os.makedirs(conf.DOWNLOAD_DIR, exist_ok=True)
# -----------------

service = build("drive", "v3", credentials=conf.CREDENTIALS)

def list_files_in_folder(folder_id):
    files = []
    page_token = None

    while True:
        response = service.files().list(
            q=f"'{folder_id}' in parents",
            fields="nextPageToken, files(id, name)",
            pageSize=1000,
            pageToken=page_token
        ).execute()

        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken", None)

        if page_token is None:
            break

    return files

def download_file(file_id, filename):
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(os.path.join(conf.DOWNLOAD_DIR, filename), "wb")
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()


if __name__ == "__main__":
    files = list_files_in_folder(conf.FOLDER_ID)
    print(f"Znaleziono {len(files)} plików.")

    for f in files:
        print("Pobieram:", f["name"])
        download_file(f["id"], f["name"])

    print("Gotowe!")
