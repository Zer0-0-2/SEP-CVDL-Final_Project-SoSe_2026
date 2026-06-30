"""Dataset - Dataset for loading breed images from disk.

Directory layout expected under root:
    raw/
        Abyssinian/
            img001.jpg
            img002.jpg
        Bengal/
            ...
        ...
Optionally also loads confounder images (label = -1) from a seperate directory
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# class list as seen in Inference.py

CLASSES = [
    "Abyssinian",        # 0
    "Bengal",            # 1
    "Birman",            # 2
    "Bombay",            # 3
    "British_Shorthair", # 4
    "Maine_Coon",        # 5
    "Ragdoll",           # 6
    "Sphynx",            # 7
    "Tabby",             # 8
    "Tiger_Cat",         # 9
    "Beagle",            # 10
    "Pug",               # 11
    "Boxer",             # 12
    "Shiba_Inu",         # 13
    "Samoyed",           # 14
    "Golden_Retriever",  # 15
    "German_Shepherd",   # 16
    "Siberian_Husky",    # 17
    "Dalmatian",         # 18
    "Rottweiler",        # 19
]

CLASS_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}
CONFOUNDER_LABEL =-1
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

class AnimalDataset(Dataset):
    """Loads breed images from per-class subdirectories and applies albumentations transforms.

    Args:
        root:       Path to the raw data directory (one subfolder per breed).
        transform:  Albumentations Compose pipeline. 
                    Called as transform(image=numpy_hwc_uint8)["image"]->CHW tensor.

        include_confounders: Also load images from confounder_dir, labelled -1.
        confounder_dir: Path to confounder images (requred when include_confounders=True).                        
    """

    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable] = None,
        include_confounders: bool = False,
        confounder_dir: Optional[str|Path] = None,
    ):
        self.root = Path(root)
        self.transform = transform
        #Each entry is (absolute_image_poath, integer_label)
        self.samples: list[tuple[Path, int]] = []
        self._build_index(include_confounders, confounder_dir)

#index building

    def _build_index(
        self,
        include_confounders: bool,
        confounder_dir:Optional[str | Path],
    ) -> None:
        """Walk self.root and populate self.samples with (path, label) pairs"""

        if not self.root.exists() or not self.root.is_dir():
            raise FileNotFoundError(f"Data root not found: {self.root}")

        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            label = CLASS_TO_IDX.get(class_dir.name)
            if label is None:
                print(f"Warning: skipping unknown folder '{class_dir.name}'")
                continue
            for file in sorted(class_dir.iterdir()):
                if file.suffix.lower() in VALID_EXTENSIONS:
                    self.samples.append((file, label))

        if include_confounders and confounder_dir is not None:
            confounder_dir = Path(confounder_dir)
            for file in sorted(confounder_dir.iterdir()):
                if file.suffix.lower() in VALID_EXTENSIONS:
                    self.samples.append((file, CONFOUNDER_LABEL))

        print(f"Dataset: {len(self.samples)} images, {len(set(l for _, l in self.samples))} classes")
    
#Dataset Interface

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Load one image and its label, apply transform, return (tensor, label)."""
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        arr = np.array(image)
        if self.transform is not None:
            arr = self.transform(image=arr)["image"]
        else:
            arr = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return arr, label


#Utility

    def class_counts(self) -> dict[str, int]:
        """Return a dict of {class_name: image_count} for logging/debugging."""

        from collections import Counter
        idx_to_name = {v: k for k, v in CLASS_TO_IDX.items()}
        idx_to_name[CONFOUNDER_LABEL] = "confounder"
        counts: Counter = Counter(label for _, label in self.samples)
        return {idx_to_name.get(k, str(k)): v for k, v in sorted(counts.items())}
    



