"""Quick sanity check for AnimalDataset and augmentations.

Run from the project root:
    python test_dataset.py

All assertions must pass before moving on to trainer.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from animal_recognition.src.data.dataset import AnimalDataset, CLASSES
from animal_recognition.src.data.augmentations import get_train_transforms, get_val_transforms

RAW_DIR = Path("animal_recognition/data/raw")
CONFOUNDER_DIR = Path("animal_recognition/data/confounders")


def test_dataset_loads():
    ds = AnimalDataset(root=RAW_DIR)
    assert len(ds) > 0, "Dataset is empty — check data/raw/ folder"
    print(f"  PASS  len(dataset) = {len(ds)}")


def test_class_counts():
    ds = AnimalDataset(root=RAW_DIR)
    counts = ds.class_counts()
    assert len(counts) == len(CLASSES), f"Expected {len(CLASSES)} classes, got {len(counts)}"
    print(f"  PASS  class counts: {counts}")


def test_getitem_shape():
    transform = get_val_transforms(224)
    ds = AnimalDataset(root=RAW_DIR, transform=transform)
    tensor, label = ds[0]
    assert tensor.shape == (3, 224, 224), f"Expected (3,224,224), got {tensor.shape}"
    assert isinstance(label, int), f"Label should be int, got {type(label)}"
    assert 0 <= label <= 19, f"Label out of range: {label}"
    print(f"  PASS  tensor shape={tensor.shape}, label={label}")


def test_tensor_normalised():
    transform = get_val_transforms(224)
    ds = AnimalDataset(root=RAW_DIR, transform=transform)
    tensor, _ = ds[0]
    assert tensor.dtype == torch.float32, f"Expected float32, got {tensor.dtype}"
    # After ImageNet normalisation values will be roughly in [-2.5, 2.5]
    assert tensor.min() < 0, "Tensor should have negative values after normalisation"
    print(f"  PASS  dtype={tensor.dtype}, range=[{tensor.min():.2f}, {tensor.max():.2f}]")


def test_train_transform_randomness():
    transform = get_train_transforms(224)
    ds = AnimalDataset(root=RAW_DIR, transform=transform)
    t1, _ = ds[0]
    t2, _ = ds[0]
    # Two passes on the same image with random augmentations should differ
    assert not torch.equal(t1, t2), "Train transforms should be random — got identical tensors"
    print(f"  PASS  train transforms produce different outputs on same image")


def test_confounders():
    ds = AnimalDataset(
        root=RAW_DIR,
        include_confounders=True,
        confounder_dir=CONFOUNDER_DIR,
    )
    labels = [label for _, label in ds.samples]
    assert -1 in labels, "No confounder samples found (label=-1)"
    confounder_count = labels.count(-1)
    print(f"  PASS  {confounder_count} confounder samples loaded with label=-1")


if __name__ == "__main__":
    tests = [
        test_dataset_loads,
        test_class_counts,
        test_getitem_shape,
        test_tensor_normalised,
        test_train_transform_randomness,
        test_confounders,
    ]
    print(f"Running {len(tests)} tests...\n")
    failed = 0
    for test in tests:
        try:
            print(f"[ {test.__name__} ]")
            test()
        except Exception as e:
            print(f"  FAIL  {e}")
            failed += 1
        print()

    if failed == 0:
        print(f"All {len(tests)} tests passed.")
    else:
        print(f"{failed}/{len(tests)} tests FAILED.")
        sys.exit(1)
