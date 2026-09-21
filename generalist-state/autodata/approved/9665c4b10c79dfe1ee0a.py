import logging
import os

from dotenv import load_dotenv

load_dotenv()

GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")
EXTRACTION_BUFFER_PATH = os.getenv("EXTRACTION_BUFFER_PATH")
TARGET_REPO_NAME = (os.getenv("GITHUB_TARGET_REPO") or "").strip()
BUFFER_PATH = os.getenv("BUFFER_PATH")
GIT_AUTHOR_NAME = os.getenv("GIT_AUTHOR_NAME", GITHUB_USERNAME)
GIT_AUTHOR_EMAIL = os.getenv(
    "GIT_AUTHOR_EMAIL", f"{GITHUB_USERNAME}@users.noreply.github.com"
)

BASE_URL = "https://api.normattiva.it/t/normattiva.api/bff-opendata/v1/api/v1"
ENDPOINT_URL = f"{BASE_URL}/collections/download/collection-preconfezionata"
COLLECTIONS_URL = f"{BASE_URL}/collections/collection-predefinite"

# Normattiva può impiegare molto per pacchetti grandi; breve connect, read lungo.
DOWNLOAD_TIMEOUT = (30.0, 300.0)
DOWNLOAD_MAX_ATTEMPTS = 3
DOWNLOAD_RETRY_SLEEP_SEC = 5.0

_level_name = os.getenv("LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _level_name, logging.INFO),
    format="%(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("italia_corpus")
