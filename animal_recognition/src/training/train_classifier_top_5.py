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


def main():

    # Suggested during Q&A by Johannes: https://blog.eleuther.ai/mutransfer/

    # 1. GCViT Base - BitFit (Biases & Norms Only)
    # muP LR Scaling: NOT APPLIED (Weight matrices are frozen)
    model_1 = GCViTClassifier(pretrained=True, model_name="gcvit_base")
    for name, param in model_1.named_parameters():
        if "bias" not in name and "head" not in name and "norm" not in name:
            param.requires_grad = False

    optimizer_1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model_1.parameters()), lr=1e-3, weight_decay=0
    )
    scheduler_1 = cosine_lr.CosineLRScheduler(
        optimizer_1, t_initial=140, lr_min=1e-5, warmup_t=10, warmup_lr_init=1e-4
    )

    trainer_1 = ClassifierTrainer(
        model=model_1,
        optimizer=optimizer_1,
        scheduler=scheduler_1,
        batch_size=16,
        augmentation_file="stronger",
    )
    trainer_1.train(epochs=150, note="gcvit_base_bitfit")

    del model_1, optimizer_1, scheduler_1, trainer_1
    torch.cuda.empty_cache()

    # 2. GCViT Base - Conservative Full Finetuning
    # muP LR Scaling: 0.5 (Base LR: 1e-5 -> 5e-6)
    model_2 = GCViTClassifier(pretrained=True, model_name="gcvit_base")

    optimizer_2 = optim.AdamW(model_2.parameters(), lr=5e-6, weight_decay=5e-4)
    scheduler_2 = cosine_lr.CosineLRScheduler(
        optimizer_2, t_initial=140, lr_min=5e-8, warmup_t=10, warmup_lr_init=5e-7
    )

    trainer_2 = ClassifierTrainer(
        model=model_2,
        optimizer=optimizer_2,
        scheduler=scheduler_2,
        batch_size=16,
        augmentation_file="stronger",
    )
    trainer_2.train(epochs=150, note="gcvit_base_conservative_finetune")

    del model_2, optimizer_2, scheduler_2, trainer_2
    torch.cuda.empty_cache()

    # 3. GCViT Base Paper Repllication

    # actual config from the fine grained cat classification paper
    model_3 = GCViTClassifier(pretrained=True, model_name="gcvit_base")

    optimizer_3 = optim.AdamW(model_3.parameters(), lr=5e-5, weight_decay=1e-4)

    # Step-wise decay, e.g., decay by 0.5 every 30 epochs

    scheduler_3 = StepLRScheduler(optimizer_3, decay_t=30, decay_rate=0.5)

    trainer_3 = ClassifierTrainer(
        model=model_3,
        optimizer=optimizer_3,
        scheduler=scheduler_3,
        batch_size=16,
        label_smoothing=0.1,
        augmentation_file="stronger",
    )

    trainer_3.train(epochs=100, patience=5, note="gcvit_base_paper_rep")

    del model_3, optimizer_3, scheduler_3, trainer_3
    torch.cuda.empty_cache()

    # 4. GCViT Base - Agressive Finetuning

    # larger LR = aggressive finetuning
    model_4 = GCViTClassifier(pretrained=True, model_name="gcvit_base")

    # Apply muP scaling (0.5) to the base learning rate
    optimizer_4 = optim.AdamW(model_4.parameters(), lr=2.5e-4, weight_decay=1e-4)

    # Apply muP scaling (0.5) to lr_min and warmup_lr_init
    scheduler_4 = cosine_lr.CosineLRScheduler(
        optimizer_4, t_initial=140, lr_min=5e-7, warmup_t=10, warmup_lr_init=2.5e-5
    )

    trainer_4 = ClassifierTrainer(
        model=model_4,
        optimizer=optimizer_4,
        scheduler=scheduler_4,
        batch_size=16,  # Reduced for Base model VRAM safety
        augmentation_file="stronger",
    )
    trainer_4.train(epochs=150, note="gcvit_base_aggressive_finetune")

    del model_4, optimizer_4, scheduler_4, trainer_4
    torch.cuda.empty_cache()

    # 5. ConvNeXtV2 Base - Conservative Finetuning

    # muP LR Scaling: 0.75 (Base LR: 1e-5 -> 7.5e-6)
    model_5 = ConvNextClassifier(pretrained=True, model_name="convnextv2_base")

    optimizer_5 = optim.AdamW(model_5.parameters(), lr=7.5e-6, weight_decay=5e-4)
    scheduler_5 = cosine_lr.CosineLRScheduler(
        optimizer_5, t_initial=140, lr_min=7.5e-8, warmup_t=10, warmup_lr_init=7.5e-7
    )
    trainer_5 = ClassifierTrainer(
        model=model_5,
        optimizer=optimizer_5,
        scheduler=scheduler_5,
        batch_size=16,
        augmentation_file="stronger",
    )
    trainer_5.train(epochs=150, note="convnextv2_base_conservative_finetune")

    del model_5, optimizer_5, scheduler_5, trainer_5
    torch.cuda.empty_cache()

    # ConvNeXtV2 Base - FROM RANDOM WEIGHTS

    model_base_scratch = ConvNextClassifier(pretrained=False, model_name="convnextv2_base")

    # separate parameters for weight decay
    # https://arxiv.org/pdf/2301.00808
    decay_base = []
    no_decay_base = []
    for name, param in model_base_scratch.named_parameters():
        if not param.requires_grad:
            continue
        # Exclude GRN gamma (weight) and beta (bias) from weight decay
        if "grn" in name and (
            "weight" in name or "bias" in name or "gamma" in name or "beta" in name
        ):
            no_decay_base.append(param)
        else:
            decay_base.append(param)

    # weight decay between 1e-4 and 1e-3
    # muP LR Scaling: 0.75 (Base LR: 1e-4 -> 7.5e-5)
    optimizer_base = optim.AdamW(
        [
            {"params": decay_base, "weight_decay": 5e-4},
            {"params": no_decay_base, "weight_decay": 0.0},
        ],
        lr=7.5e-5,
    )

    # https://timm.fast.ai/SGDR
    # muP LR Scaling applied to min_lr and warmup_lr_init
    scheduler_base = cosine_lr.CosineLRScheduler(
        optimizer_base,
        t_initial=140,  # number of epochs PER CYCLE -_-
        lr_min=7.5e-7,
        warmup_t=10,
        warmup_lr_init=7.5e-6,
        warmup_prefix=True,
    )

    trainer_scratch_optimized_base = ClassifierTrainer(
        model=model_base_scratch,
        optimizer=optimizer_base,
        scheduler=scheduler_base,
        batch_size=16,
        label_smoothing=0.2,
        image_size=224,
        augmentation_file="vetted",
    )

    trainer_scratch_optimized_base.train(
        epochs=150,
        note="cosinelr_with_warmup_optimized_weight_decay_base_scratch",
    )


if __name__ == "__main__":
    main()
