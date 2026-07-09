import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import animal_recognition.src.data.augmentations as augmentations
import animal_recognition.src.data.augmentations_mild as augmentations_mild
import animal_recognition.src.data.augmentations_vetted as augmentations_vetted
from animal_recognition.src.data.dataset import AnimalDataset
from animal_recognition.src.models.classifier_convnext import ConvNextClassifier
from animal_recognition.src.models.classifier_gcvit import GCViTClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "animal_recognition" / "data" / "processed" / "accepted"
DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "animal_recognition" / "models" / "weights"


class ClassifierTrainer:
    def __init__(
        self,
        architecture: str,
        model_name: str,
        data_dir: Path = DEFAULT_DATA_DIR,
        pretrained: bool = True,
        batch_size: int = 32,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.0,
        image_size: int = 224,
        augmentation_file: str = "vetted",
    ):
        """
        architecture: "convnext" or "gcvit"
        model_name: see classifier_convnext.py or classifier_gcvit.py for timm output
        """
        self.architecture = architecture
        self.data_dir = data_dir
        self.model_name = model_name
        self.pretrained = pretrained
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.image_size = image_size
        self.augmentation_file = augmentation_file
        self.device = torch.device(
            torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else "cpu"
        )

        DEFAULT_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

        print(f"Using device: {self.device}")
        print(f"Data Directory: {self.data_dir}")
        print(
            f"Architecture: {self.architecture} | Model: {self.model_name} (Pretrained: {self.pretrained})"
        )

        if self.architecture == "convnext":
            self.model = ConvNextClassifier(pretrained=self.pretrained, model_name=self.model_name)
        elif self.architecture == "gcvit":
            self.model = GCViTClassifier(pretrained=self.pretrained, model_name=self.model_name)
        else:
            raise ValueError(f"Wrong architecture: {self.architecture}")

        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def get_transforms(self):
        aug_modules = {
            "mild": augmentations_mild,
            "base": augmentations,
            "vetted": augmentations_vetted,
        }

        if self.augmentation_file not in aug_modules:
            raise ValueError(f"Unknown augmentation strategy: '{self.augmentation_file}'")

        aug_module = aug_modules[self.augmentation_file]

        train_transform = aug_module.get_train_transforms(image_size=self.image_size)

        val_transform = aug_module.get_val_transforms(image_size=self.image_size)

        return train_transform, val_transform

    def setup_dataloaders(self):
        train_transform, val_transform = self.get_transforms()

        # Create two dataset instances with DIFFERENT transforms
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

    def train(self, epochs: int = 10, patience: int = 15, note: str = ""):

        train_loader, val_loader = self.setup_dataloaders()

        # Track both best accuracy (for saving) and best loss (for stopping)
        best_val_acc = 0.0
        best_val_loss = float("inf")
        epochs_no_improve = 0

        # for plotting
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}

        params_str = f"pre{str(self.pretrained)}_bs{self.batch_size}_lr{self.lr}_wd{self.weight_decay}_ls{self.label_smoothing}_sz{self.image_size}_aug{self.augmentation_file}"
        save_path = (
            DEFAULT_WEIGHTS_DIR / f"{self.architecture}_{self.model_name}_{note}_{params_str}.pt"
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)

        last_epoch = 0

        for epoch in range(1, epochs + 1):
            last_epoch = epoch
            print(f"\nEpoch {epoch}/{epochs}")

            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["lr"].append(current_lr)

            print(
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | LR: {current_lr:.6f}"
            )
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(
                    f"New best validation accuracy ({best_val_acc:.4f})! Saving model to {save_path}"
                )
                torch.save(self.model.state_dict(), save_path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"Early Stopping Counter: {epochs_no_improve} out of {patience}")

            if epochs_no_improve >= patience:
                print(
                    f"\nEarly stopping triggered. Validation loss has not improved in {epochs_no_improve} epochs."
                )
                break

        self.create_plot(history, epochs_run=last_epoch, note=note, params_str=params_str)

    def create_plot(self, history, epochs_run, note, params_str):
        results_dir = PROJECT_ROOT / "animal_recognition" / "src" / "training" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            results_dir
            / f"training_metrics_{self.architecture}_{self.model_name}_{note}_{params_str}_{timestamp}.png"
        )

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

        epochs_range = range(1, epochs_run + 1)

        # Loss
        ax1.plot(epochs_range, history["train_loss"], label="Train Loss")
        ax1.plot(epochs_range, history["val_loss"], label="Val Loss")
        ax1.set_title("Loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend()

        # Accuracy
        ax2.plot(epochs_range, history["train_acc"], label="Train Accuracy")
        ax2.plot(epochs_range, history["val_acc"], label="Val Accuracy")
        ax2.set_title("Accuracy")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Accuracy")
        ax2.legend()

        # Learning Rate
        ax3.plot(epochs_range, history["lr"], label="Learning Rate", color="orange")
        ax3.set_title("Learning Rate")
        ax3.set_xlabel("Epochs")
        ax3.set_ylabel("LR")
        ax3.legend()

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Training graphs saved to {filename}")


if __name__ == "__main__":
    trainer_scratch_optimized = ClassifierTrainer(
        architecture="convnext",
        model_name="convnext_small",
        pretrained=False,
        batch_size=32,
        lr=3e-3,
        weight_decay=0.05,
        label_smoothing=0.2,
        image_size=224,
        augmentation_file="vetted",
    )

    trainer_scratch_optimized.train(
        epochs=150,
        note="aug_val_warmup",
    )
