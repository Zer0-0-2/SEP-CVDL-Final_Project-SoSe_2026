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
from animal_recognition.src.models.classifier_gcvit import GCViTClassifier
from animal_recognition.src.data.augmentations_mild import get_train_transforms, get_val_transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "animal_recognition" / "data" / "processed" / "accepted"
DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "animal_recognition" / "models" / "weights"


# https://github.com/NVlabs/GCVit
class GCViTTrainer:
    def __init__(
        self,
        data_dir: Path = DEFAULT_DATA_DIR,
        model_name: str = "gcvit_tiny",
        pretrained: bool = True,
        batch_size: int = 32,
        lr: float = 0.01,
        image_size: int = 224,
    ):
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

        self.model = GCViTClassifier(pretrained=self.pretrained, model_name=self.model_name)
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

    def get_transforms(self):
        train_transform = get_train_transforms(image_size=self.image_size)
        val_transform = get_val_transforms(image_size=self.image_size)
        return train_transform, val_transform

    def setup_dataloaders(self):
        train_transform, val_transform = self.get_transforms()

        full_dataset_train = AnimalDataset(self.data_dir, transform=train_transform)
        full_dataset_val = AnimalDataset(self.data_dir, transform=val_transform)

        total_size = len(full_dataset_train)
        if total_size == 0:
            raise ValueError(f"No images found in {self.data_dir}. Run sanitize_scraped_data.py")

        val_size = int(0.2 * total_size)
        train_size = total_size - val_size

        generator = torch.Generator().manual_seed(67)
        indices = torch.randperm(total_size, generator=generator).tolist()

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_dataset = Subset(full_dataset_train, train_indices)
        val_dataset = Subset(full_dataset_val, val_indices)

        print(f"Dataset split: {train_size} training images, {val_size} validation images")

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
        # https://tqdm.github.io/
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
    trainer = GCViTTrainer(
        model_name="gcvit_tiny",
        pretrained=True,
        batch_size=32,
        lr=1e-2,
        image_size=224,
    )
    trainer.train(epochs=10)
