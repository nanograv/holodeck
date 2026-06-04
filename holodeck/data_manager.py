import os
import logging
from pathlib import Path
from importlib import resources
import pooch

logger = logging.getLogger("holodeck.data")
CACHE_DIR = Path(__file__).parent / "data"

RECORD_ID = "20534588" 
REMOTE_BASE_URL = f"https://zenodo.org/records/{RECORD_ID}/files/"


DATA_FETCH_ENGINE = pooch.create(
    path=CACHE_DIR,
    base_url=REMOTE_BASE_URL,
    env="HOLODECK_DATA_DIR",
    registry=None,
)

REGISTRY_PATH = CACHE_DIR / "registry.txt"
try:
    with open(REGISTRY_PATH, "r") as f:
        DATA_FETCH_ENGINE.load_registry(f)
except FileNotFoundError:
    logger.warning(f"Data registry file not found at {REGISTRY_PATH}. On-demand downloads will fail.")


def get_data_path(relative_path: str) -> Path:
    """ Resolves the path using Pooch. Downloads if missing. """
    try:
        return Path(DATA_FETCH_ENGINE.fetch(relative_path))
    except Exception as e:
        raise FileNotFoundError(f"Could not resolve asset '{relative_path}': {e}")