import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from animal_recognition.src.data.dataset import AnimalDataset
from animal_recognition.src.models.classifier_convnext import ConvNextClassifier
import animal_recognition.src.data.augmentations_mild as augmentations_mild
import animal_recognition.src.data.augmentations as augmentations_strong


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "animal_recognition" / "data" / "processed" / "accepted"
DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "animal_recognition" / "models" / "weights"


class ConvNextTrainer:
    def __init__(
        self,
        data_dir: Path = DEFAULT_DATA_DIR,
        model_name: str = "convnext_tiny",
        pretrained: bool = True,
        batch_size: int = 32,
        lr: float = 0.01,
        image_size: int = 224,
    ):
        """
        Model names: "convnext_tiny", "convnext_small", "convnext_base", "convnext_large"
        the first three have 224x224 input size, the last one has 384x384 input size (probably too large for compute resources)
        """
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.pretrained = pretrained
        self.batch_size = batch_size
        self.lr = lr
        self.image_size = image_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        DEFAULT_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Using device: {self.device}")
        print(f"Data Directory: {self.data_dir}")
        print(f"Model: {self.model_name} (Pretrained: {self.pretrained})")

        self.model = ConvNextClassifier(pretrained=self.pretrained, model_name=self.model_name)
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

    def get_transforms(self):
        train_transform = augmentations_mild.get_train_transforms(image_size=self.image_size)
        val_transform = augmentations_mild.get_val_transforms(image_size=self.image_size)

        return train_transform, val_transform

    def setup_dataloaders(self):
        train_transform, val_transform = self.get_transforms()

        # Create two dataset instances with DIFFERENT transforms
        full_dataset_train = AnimalDataset(self.data_dir, transform=train_transform)
        full_dataset_val = AnimalDataset(self.data_dir, transform=val_transform)

        total_size = len(full_dataset_train)

        val_size = int(0.2 * total_size)
        train_size = total_size - val_size

        generator = torch.Generator().manual_seed(67)
        indices = torch.randperm(total_size, generator=generator).tolist()

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_dataset = Subset(full_dataset_train, train_indices)
        val_dataset = Subset(full_dataset_val, val_indices)

        print(f"Dataset split: {train_size} training images, {val_size} validation images")

        # persisten_workers = True to avoid no such file or directory error
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        return train_loader, val_loader

    def train_one_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Progress bar, techniaclly not needed
        pbar = tqdm(dataloader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({"loss": loss.item(), "acc": correct / total})

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        ## https://tqdm.github.io/

        ## simlar to https://adamoudad.github.io/posts/progress_bar_with_tqdm/
        pbar = tqdm(dataloader, desc="Validation")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def train(self, epochs: int = 10):

        train_loader, val_loader = self.setup_dataloaders()
        best_val_acc = 0.0
        save_path = (
            DEFAULT_WEIGHTS_DIR
            / f"{self.model_name}_{str(self.pretrained)}_{self.batch_size}_{self.lr}_{self.image_size}.pt"
        )

        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")

            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(
                    f"New best validation accuracy ({best_val_acc:.4f})! Saving model to {save_path}"
                )
                torch.save(self.model.state_dict(), save_path)


if __name__ == "__main__":
    trainer_baseline = ConvNextTrainer(
        model_name="convnext_tiny",
        pretrained=False,
        batch_size=32,
        lr=1e-3,
        image_size=224,
    )
    trainer_baseline.train(epochs=80)

    trainer_pretrained = ConvNextTrainer(
        model_name="convnext_tiny",
        pretrained=True,
        batch_size=32,
        lr=1e-4,
        image_size=224,
    )
    trainer_pretrained.train(epochs=15)

    trainer_highres = ConvNextTrainer(
        model_name="convnext_small",
        pretrained=True,
        batch_size=16,  # not enough vram :/
        lr=5e-5,
        image_size=384,
    )
    trainer_highres.train(epochs=15)

    trainer_lowres = ConvNextTrainer(
        model_name="convnext_tiny",
        pretrained=True,
        batch_size=64,
        lr=2e-4,
        image_size=128,
    )
    trainer_lowres.train(epochs=5)
