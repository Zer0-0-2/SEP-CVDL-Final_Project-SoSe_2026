import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Import the specific scheduler class directly
from timm.scheduler import cosine_lr
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import animal_recognition.src.data.augmentations as augmentations
import animal_recognition.src.data.augmentations_mild as augmentations_mild
import animal_recognition.src.data.augmentations_stronger as augmentations_stronger
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
        model,
        data_dir: Path = DEFAULT_DATA_DIR,
        batch_size: int = 32,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.0,
        image_size: int = 224,
        augmentation_file: str = "vetted",
        optimizer=None,
        scheduler=None,
    ):
        """
        architecture: "convnext" or "gcvit"
        model_name: see classifier_convnext.py or classifier_gcvit.py for timm output
        """

        self.data_dir = data_dir
        self.architecture = model.architecture  # gcvit or convnext
        self.model_name = model.model_name  # specific name
        self.pretrained = model.pretrained  # True/false
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.image_size = image_size
        self.augmentation_file = augmentation_file
        self.scheduler = scheduler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_name = (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else "cpu"
        )

        DEFAULT_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

        print(f"Using device: {self.device_name}")
        print(f"Data Directory: {self.data_dir}")
        print(
            f"Architecture: {self.architecture} | Model: {self.model_name} (Pretrained: {self.pretrained})"
        )
        self.model = model

        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        if optimizer is not None:
            self.optimizer = optimizer
            # Try extracting these to keep filename string consistent
            self.lr = self.optimizer.param_groups[0].get("lr", self.lr)
            self.weight_decay = self.optimizer.param_groups[0].get(
                "weight_decay", self.weight_decay
            )
        else:
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

        self.scheduler = scheduler

    def get_transforms(self):
        aug_modules = {
            "mild": augmentations_mild,
            "base": augmentations,
            "vetted": augmentations_vetted,
            "stronger": augmentations_stronger,
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
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
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

            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
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

        # TODO: fix this somewhat whacky implementation, works for now
        if self.scheduler is not None:
            scheduler = self.scheduler
        else:
            scheduler = None

        params_str = f"pre{str(self.pretrained)}_bs{self.batch_size}_lr{self.lr}_wd{self.weight_decay}_ls{self.label_smoothing}_sz{self.image_size}_aug{self.augmentation_file}_sched{self.scheduler.__class__.__name__ if self.scheduler is not None else 'None'}"

        print(f"Training with parameters: {params_str}")
        save_path = DEFAULT_WEIGHTS_DIR / f"{self.model_name}_{note}_{params_str}.pt"

        last_epoch = 0

        for epoch in range(1, epochs + 1):
            last_epoch = epoch
            print(f"\nEpoch {epoch}/{epochs}")

            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            # Step the timm scheduler with the epoch number
            if scheduler is not None:
                scheduler.step(epoch)

            # Extract LR straight from the optimizer parameter groups
            current_lr = self.optimizer.param_groups[0]["lr"]

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
    # use this more compact and modular syntax from now on
    """
    model_0 = ConvNextClassifier(pretrained=False, model_name="convnextv2_tiny")

    optimizer_0 = optim.AdamW(model_0.parameters(), lr=1e-4, weight_decay=5e-4)

    # https://timm.fast.ai/SGDR
    scheduler_0 = cosine_lr.CosineLRScheduler(
        optimizer_0,
        t_initial=140,  # number of epochs PER CYCLE -_-
        lr_min=1e-6,
        warmup_t=10,
        warmup_lr_init=1e-5,
    )

    trainer_scratch_optimized_0 = ClassifierTrainer(
        model=model_0,
        optimizer=optimizer_0,
        scheduler=scheduler_0,
        batch_size=32,
        label_smoothing=0.2,
        image_size=224,
        augmentation_file="vetted",
    )

    trainer_scratch_optimized_0.train(
        epochs=150,
        note="cosinelr_with_warmup",
    )

    # delete after run and before
    del model_0, optimizer_0, scheduler_0, trainer_scratch_optimized_0
    torch.cuda.empty_cache()

    model_1 = ConvNextClassifier(pretrained=False, model_name="convnextv2_tiny")

    # separate parameters for weight decay
    # https://arxiv.org/pdf/2301.00808
    decay_1 = []
    no_decay_1 = []
    for name, param in model_1.named_parameters():
        if not param.requires_grad:
            continue
        # Exclude GRN gamma (weight) and beta (bias) from weight decay
        if "grn" in name and (
            "weight" in name or "bias" in name or "gamma" in name or "beta" in name
        ):
            no_decay_1.append(param)
        else:
            decay_1.append(param)

    # weight decay between 1e-4 and 1e-3
    optimizer_1 = optim.AdamW(
        [{"params": decay_1, "weight_decay": 5e-4}, {"params": no_decay_1, "weight_decay": 0.0}],
        lr=1e-4,
    )

    # https://timm.fast.ai/SGDR
    scheduler_1 = cosine_lr.CosineLRScheduler(
        optimizer_1,
        t_initial=140,  # number of epochs PER CYCLE -_-
        lr_min=1e-6,
        warmup_t=10,
        warmup_lr_init=1e-5,
        warmup_prefix=True,
    )

    trainer_scratch_optimized_1 = ClassifierTrainer(
        model=model_1,
        optimizer=optimizer_1,
        scheduler=scheduler_1,
        batch_size=32,
        label_smoothing=0.2,
        image_size=224,
        augmentation_file="vetted",
    )

    trainer_scratch_optimized_1.train(
        epochs=150,
        note="cosinelr_with_warmup_optimized_weight_decay",
    )

    del model_1, optimizer_1, scheduler_1, trainer_scratch_optimized_1
    torch.cuda.empty_cache()
    model_2 = ConvNextClassifier(pretrained=False, model_name="convnextv2_tiny")

    optimizer_2 = optim.AdamW(model_2.parameters(), lr=1e-4, weight_decay=5e-4)

    # https://timm.fast.ai/SGDR
    scheduler_2 = cosine_lr.CosineLRScheduler(
        optimizer_2,
        t_initial=140,  # number of epochs PER CYCLE -_-
        lr_min=1e-6,
        warmup_t=10,
        warmup_lr_init=1e-5,
    )

    trainer_scratch_optimized_2 = ClassifierTrainer(
        model=model_2,
        optimizer=optimizer_2,
        scheduler=scheduler_2,
        batch_size=32,
        label_smoothing=0.2,
        image_size=224,
        augmentation_file="stronger",
    )

    trainer_scratch_optimized_2.train(
        epochs=150,
        note="cosinelr_with_warmup",
    )

    # delete after run and before
    del model_2, optimizer_2, scheduler_2, trainer_scratch_optimized_2
    torch.cuda.empty_cache()

    model_3 = ConvNextClassifier(pretrained=False, model_name="convnextv2_tiny")

    # separate parameters for weight decay
    # https://arxiv.org/pdf/2301.00808
    decay_3 = []
    no_decay_3 = []
    for name, param in model_3.named_parameters():
        if not param.requires_grad:
            continue
        # Exclude GRN gamma (weight) and beta (bias) from weight decay
        if "grn" in name and (
            "weight" in name or "bias" in name or "gamma" in name or "beta" in name
        ):
            no_decay_3.append(param)
        else:
            decay_3.append(param)

    # weight decay between 1e-4 and 1e-3
    optimizer_3 = optim.AdamW(
        [{"params": decay_3, "weight_decay": 5e-4}, {"params": no_decay_3, "weight_decay": 0.0}],
        lr=1e-4,
    )

    # https://timm.fast.ai/SGDR
    scheduler_3 = cosine_lr.CosineLRScheduler(
        optimizer_3,
        t_initial=140,  # number of epochs PER CYCLE -_-
        lr_min=1e-6,
        warmup_t=10,
        warmup_lr_init=1e-5,
        warmup_prefix=True,
    )

    trainer_scratch_optimized_3 = ClassifierTrainer(
        model=model_3,
        optimizer=optimizer_3,
        scheduler=scheduler_3,
        batch_size=32,
        label_smoothing=0.2,
        image_size=224,
        augmentation_file="stronger",
    )

    trainer_scratch_optimized_3.train(
        epochs=150,
        note="cosinelr_with_warmup_optimized_weight_decay",
    )
    del model_3, optimizer_3, scheduler_3, trainer_scratch_optimized_3
    torch.cuda.empty_cache()
    """

    # GCViT (normal finetuning, nothing special)
    model_4 = GCViTClassifier(pretrained=True, model_name="gcvit_tiny")

    optimizer_4 = optim.AdamW(model_4.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler_4 = cosine_lr.CosineLRScheduler(
        optimizer_4, t_initial=140, lr_min=1e-6, warmup_t=10, warmup_lr_init=1e-5
    )

    trainer_4 = ClassifierTrainer(
        model=model_4,
        optimizer=optimizer_4,
        scheduler=scheduler_4,
        batch_size=32,
        augmentation_file="vetted",
    )
    trainer_4.train(epochs=150, note="gcvit_tiny_standard_finetune")

    del model_4, optimizer_4, scheduler_4, trainer_4
    torch.cuda.empty_cache()

    # smaller LR = conservative finetuning
    model_5 = GCViTClassifier(pretrained=True, model_name="gcvit_tiny")

    optimizer_5 = optim.AdamW(model_5.parameters(), lr=1e-5, weight_decay=5e-4)
    scheduler_5 = cosine_lr.CosineLRScheduler(
        optimizer_5, t_initial=140, lr_min=1e-7, warmup_t=10, warmup_lr_init=1e-6
    )

    trainer_5 = ClassifierTrainer(
        model=model_5,
        optimizer=optimizer_5,
        scheduler=scheduler_5,
        batch_size=32,
        augmentation_file="stronger",
    )
    trainer_5.train(epochs=150, note="gcvit_tiny_conservative_finetune")

    del model_5, optimizer_5, scheduler_5, trainer_5
    torch.cuda.empty_cache()

    # larger LR = agressive finetuning
    model_6 = GCViTClassifier(pretrained=True, model_name="gcvit_tiny")

    optimizer_6 = optim.AdamW(model_6.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler_6 = cosine_lr.CosineLRScheduler(
        optimizer_6, t_initial=140, lr_min=1e-6, warmup_t=10, warmup_lr_init=5e-5
    )

    trainer_6 = ClassifierTrainer(
        model=model_6,
        optimizer=optimizer_6,
        scheduler=scheduler_6,
        batch_size=32,
        augmentation_file="vetted",
    )
    trainer_6.train(epochs=150, note="gcvit_tiny_aggressive_finetune")

    del model_6, optimizer_6, scheduler_6, trainer_6
    torch.cuda.empty_cache()

    # freeze all except the head, smaller lr because of that
    model_7 = GCViTClassifier(pretrained=True, model_name="gcvit_tiny")
    for name, param in model_7.named_parameters():
        if "head" not in name:
            param.requires_grad = False

    optimizer_7 = optim.AdamW(
        filter(lambda p: p.requires_grad, model_7.parameters()), lr=1e-3, weight_decay=1e-4
    )
    scheduler_7 = cosine_lr.CosineLRScheduler(
        optimizer_7, t_initial=140, lr_min=1e-5, warmup_t=10, warmup_lr_init=1e-4
    )

    trainer_7 = ClassifierTrainer(
        model=model_7,
        optimizer=optimizer_7,
        scheduler=scheduler_7,
        batch_size=32,
        augmentation_file="vetted",
    )
    trainer_7.train(epochs=150, note="gcvit_tiny_linear_probe")

    del model_7, optimizer_7, scheduler_7, trainer_7
    torch.cuda.empty_cache()

    # freeze head and stage 3
    model_8 = GCViTClassifier(pretrained=True, model_name="gcvit_tiny")
    for name, param in model_8.named_parameters():
        if "head" not in name and "stages.3" not in name:
            param.requires_grad = False

    optimizer_8 = optim.AdamW(
        filter(lambda p: p.requires_grad, model_8.parameters()), lr=1e-4, weight_decay=1e-4
    )
    scheduler_8 = cosine_lr.CosineLRScheduler(
        optimizer_8, t_initial=140, lr_min=1e-6, warmup_t=10, warmup_lr_init=1e-5
    )

    trainer_8 = ClassifierTrainer(
        model=model_8,
        optimizer=optimizer_8,
        scheduler=scheduler_8,
        batch_size=32,
        augmentation_file="stronger",
    )
    trainer_8.train(epochs=150, note="gcvit_tiny_partial_freeze")

    del model_8, optimizer_8, scheduler_8, trainer_8
    torch.cuda.empty_cache()

    # custom weight decay like in ConvNext
    model_9 = GCViTClassifier(pretrained=True, model_name="gcvit_tiny")
    decay_9 = []
    no_decay_9 = []
    for name, param in model_9.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay_9.append(param)
        else:
            decay_9.append(param)

    optimizer_9 = optim.AdamW(
        [{"params": decay_9, "weight_decay": 5e-4}, {"params": no_decay_9, "weight_decay": 0.0}],
        lr=1e-4,
    )
    scheduler_9 = cosine_lr.CosineLRScheduler(
        optimizer_9, t_initial=140, lr_min=1e-6, warmup_t=10, warmup_lr_init=1e-5
    )

    trainer_9 = ClassifierTrainer(
        model=model_9,
        optimizer=optimizer_9,
        scheduler=scheduler_9,
        batch_size=32,
        augmentation_file="stronger",
    )
    trainer_9.train(epochs=150, note="gcvit_tiny_custom_wd")

    del model_9, optimizer_9, scheduler_9, trainer_9
    torch.cuda.empty_cache()

    # longer warmup
    model_10 = GCViTClassifier(pretrained=True, model_name="gcvit_tiny")
    optimizer_10 = optim.AdamW(model_10.parameters(), lr=1e-4, weight_decay=1e-4)
    # 20 epochs warmup
    scheduler_10 = cosine_lr.CosineLRScheduler(
        optimizer_10, t_initial=140, lr_min=1e-6, warmup_t=20, warmup_lr_init=1e-6
    )

    trainer_10 = ClassifierTrainer(
        model=model_10,
        optimizer=optimizer_10,
        scheduler=scheduler_10,
        batch_size=32,
        augmentation_file="stronger",
    )
    trainer_10.train(epochs=150, note="gcvit_tiny_long_warmup")

    del model_10, optimizer_10, scheduler_10, trainer_10
    torch.cuda.empty_cache()

    from timm.scheduler.step_lr import StepLRScheduler

    # actual config from the fine grained cat classification paper
    model_11 = GCViTClassifier(pretrained=True, model_name="gcvit_tiny")
    optimizer_11 = optim.AdamW(model_11.parameters(), lr=1e-4, weight_decay=1e-4)
    # Step-wise decay, e.g., decay by 0.5 every 30 epochs
    scheduler_11 = StepLRScheduler(optimizer_11, decay_t=30, decay_rate=0.5)

    trainer_11 = ClassifierTrainer(
        model=model_11,
        optimizer=optimizer_11,
        scheduler=scheduler_11,
        batch_size=32,
        label_smoothing=0.1,
        augmentation_file="stronger",
    )
    trainer_11.train(epochs=100, patience=5, note="gcvit_tiny_paper_rep")

    del model_11, optimizer_11, scheduler_11, trainer_11
    torch.cuda.empty_cache()

    # bottom 50% layer freezing
    model_12 = GCViTClassifier(pretrained=True, model_name="gcvit_tiny")
    # Freeze stages 0 and 1
    for name, param in model_12.named_parameters():
        if "stages.0" in name or "stages.1" in name or "patch_embed" in name:
            param.requires_grad = False

    optimizer_12 = optim.AdamW(
        filter(lambda p: p.requires_grad, model_12.parameters()), lr=1e-4, weight_decay=1e-4
    )
    scheduler_12 = cosine_lr.CosineLRScheduler(
        optimizer_12, t_initial=140, lr_min=1e-6, warmup_t=10, warmup_lr_init=1e-5
    )

    trainer_12 = ClassifierTrainer(
        model=model_12,
        optimizer=optimizer_12,
        scheduler=scheduler_12,
        batch_size=32,
        label_smoothing=0.1,
        augmentation_file="vetted",
    )
    trainer_12.train(epochs=150, note="gcvit_tiny_bottom_freeze")

    del model_12, optimizer_12, scheduler_12, trainer_12
    torch.cuda.empty_cache()

    # custom learning rate depending on the layer (idk if that will work, sounds interesting though)
    model_13 = GCViTClassifier(pretrained=True, model_name="gcvit_tiny")

    # Assign different learning rates based on stage depth
    param_groups_13 = []
    for name, param in model_13.named_parameters():
        if not param.requires_grad:
            continue

        lr_scale = 1.0
        if "patch_embed" in name or "stages.0" in name:
            lr_scale = 0.1
        elif "stages.1" in name:
            lr_scale = 0.2
        elif "stages.2" in name:
            lr_scale = 0.5
        # stages.3 and head get 1.0 scale for now

        param_groups_13.append({"params": [param], "lr": 1e-4 * lr_scale, "weight_decay": 1e-4})

    optimizer_13 = optim.AdamW(param_groups_13)
    # Warmup base lr and min lr
    scheduler_13 = cosine_lr.CosineLRScheduler(
        optimizer_13, t_initial=140, lr_min=1e-6, warmup_t=10, warmup_lr_init=1e-5
    )

    trainer_13 = ClassifierTrainer(
        model=model_13,
        optimizer=optimizer_13,
        scheduler=scheduler_13,
        batch_size=32,
        label_smoothing=0.1,
        augmentation_file="stronger",
    )
    trainer_13.train(epochs=150, note="gcvit_tiny_layer_decay")

    del model_13, optimizer_13, scheduler_13, trainer_13
    torch.cuda.empty_cache()
