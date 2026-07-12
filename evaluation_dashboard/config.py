from pathlib import Path

# Paths
BASE_DIR = Path("/app/animal_recognition")
DATASET_DIR = BASE_DIR / "data/test_dataset"
WEIGHTS_DIR = BASE_DIR / "models" / "weights"

CACHE_DIR = Path("/app/cache_data")
CACHE_FILE = CACHE_DIR / "eval_cache.pkl"

from animal_recognition.src.data.dataset import CLASSES

# Shared Memory Cache
evaluation_cache = {}
