import logging
from pathlib import Path

# Configure global robust logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("dashboard")

# Paths
BASE_DIR = Path("/app/animal_recognition")
DATASET_DIR = BASE_DIR / "data/test_dataset"
WEIGHTS_DIR = BASE_DIR / "models" / "weights"

CACHE_DIR = Path("/app/cache_data")
CACHE_FILE = CACHE_DIR / "eval_cache.pkl"

from animal_recognition.src.data.dataset import CLASSES

# Shared Memory Cache
evaluation_cache = {}
