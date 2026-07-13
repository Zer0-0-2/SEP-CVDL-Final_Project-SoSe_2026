import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Import the specific scheduler class directly
from timm.scheduler import cosine_lr
from timm.scheduler.step_lr import StepLRScheduler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import animal_recognition.src.data.augmentations as augmentations
import animal_recognition.src.data.augmentations_mild as augmentations_mild
import animal_recognition.src.data.augmentations_stronger as augmentations_stronger
import animal_recognition.src.data.augmentations_vetted as augmentations_vetted
from animal_recognition.src.data.dataset import AnimalDataset
from animal_recognition.src.models.classifier_convnext import ConvNextClassifier
from animal_recognition.src.models.classifier_gcvit import GCViTClassifier
from animal_recognition.src.training.train_classifier import ClassifierTrainer


def start_training(model_name, lr, scheduler_type, note, warmup_t):
    print("===============================================================================")
    print(f"Starting {model_name} | LR = {lr} | {scheduler_type} | Note: {note}")
    print("===============================================================================")

    model = GCViTClassifier(pretrained=True, model_name=model_name)

    # Bitfit + layernorm
    for name, param in model.named_parameters():
        if "bias" not in name and "head" not in name and "norm" not in name:
            param.requires_grad = False

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0
    )

    if scheduler_type == "cosine":
        scheduler = cosine_lr.CosineLRScheduler(
            optimizer, t_initial=140, lr_min=lr / 100.0, warmup_t=warmup_t, warmup_lr_init=lr / 10.0
        )
    elif scheduler_type == "steplr":
        # Added warmup arguments to match your experiment notes
        scheduler = StepLRScheduler(
            optimizer, decay_t=8, decay_rate=0.5, warmup_t=warmup_t, warmup_lr_init=lr / 10.0
        )

    trainer = ClassifierTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=32,
        augmentation_file="stronger",
    )

    trainer.train(epochs=150, note=note)

    del model, optimizer, scheduler, trainer
    torch.cuda.empty_cache()


def main():
    learning_rates = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5]
    models = [
        "gcvit_base",
    ]

    for lr in learning_rates:
        for model_name in models:
            # Cosine like best model
            start_training(
                model_name=model_name,
                lr=lr,
                scheduler_type="cosine",
                note=f"{model_name}_bitfit_experiment_base_{lr}",
                warmup_t=10,
            )

            # smaller warmup
            start_training(
                model_name=model_name,
                lr=lr,
                scheduler_type="cosine",
                note=f"{model_name}_bitfit_experiment_smaller_warmup_{lr}",
                warmup_t=5,
            )

            # steplr like in cat breed paper but with smaller warmup
            start_training(
                model_name=model_name,
                lr=lr,
                scheduler_type="steplr",
                note=f"{model_name}_bitfit_experiment_smaller_warmup_steplr_{lr}",
                warmup_t=5,
            )


if __name__ == "__main__":
    main()
