from google.oauth2.service_account import Credentials

FOLDER_ID = "1MoUEp04HmohN28lhSZvzNsQydZ_yNg6m"
CREDENTIALS = Credentials.from_service_account_file("src/drive_download/service_account.json")
DOWNLOAD_DIR = "img"
