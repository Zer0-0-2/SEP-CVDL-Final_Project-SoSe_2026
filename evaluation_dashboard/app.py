import os
import pickle
from pathlib import Path

from config import CACHE_FILE, WEIGHTS_DIR, evaluation_cache, logger
from utils import get_model_list, compute_sha256
from inference import precompute_all_models_batched
from ui import create_ui, custom_css, my_theme

if __name__ == "__main__":
    CACHE_DIR = Path("/app/cache_data")
    CACHE_DIR.mkdir(exist_ok=True)
    
    logger.info("Computing SHA256 hashes for all available models...")
    models = get_model_list()
    model_hashes = {}
    for w in models:
        model_hashes[w] = compute_sha256(WEIGHTS_DIR / w)
        
    disk_cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                disk_cache = pickle.load(f)
            logger.info("Successfully read disk cache structure.")
        except Exception as e:
            logger.error(f"Failed to read disk cache: {e}")
            disk_cache = {}
            
    models_to_compute = []
    
    logger.info("Validating cache with SHA256 hashes...")
    for w in models:
        current_hash = model_hashes[w]
        
        if w in disk_cache and disk_cache[w][0] == current_hash:
            evaluation_cache[w] = disk_cache[w][1]
        else:
            basename = os.path.basename(w)
            if basename in disk_cache and disk_cache[basename][0] == current_hash:
                logger.info(f"Migrating cache entry for {basename} -> {w}")
                evaluation_cache[w] = disk_cache[basename][1]
                disk_cache[w] = disk_cache.pop(basename)
            else:
                models_to_compute.append(w)
            
    if models_to_compute:
        logger.info(f"Found {len(models_to_compute)} new or modified models requiring evaluation.")
        precompute_all_models_batched(models_to_compute, disk_cache, model_hashes, chunk_size=3)
        
        logger.info("Saving updated evaluation cache to disk...")
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(disk_cache, f)
        logger.info("Cache saved successfully!")
    else:
        logger.info("All models are cached and verified. Skipping pre-computation.")
        
    logger.info("Launching dashboard UI server...")
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, css=custom_css, theme=my_theme)
