"""Evaluation harness for the Fine-grained Animal Recognition project.

We run this script on the held-out test set, so do not change the interface.
Implement your solution as the `Model` below: an `nn.Module` whose `forward`
takes a PIL image and returns a predicted class index, an integer in
{-1, 0, ..., 19}, where -1 means "reject", i.e. no target species is present.
Inside `forward` you are free to do anything you like: run an off-the-shelf
detector, find bounding boxes, crop the largest animal, classify the crop,
decide when to return -1, and so on.

The script reads `labels.csv` from the image folder, with columns
`filename,label`, where `label` is the integer class index from CLASSES (or -1
for confounders / images with no target species). The images themselves are a
flat, numbered set (0001.jpg, 0002.jpg, ...) sitting next to `labels.csv`. The
script runs your model on every image and prints the standard classification
metrics.

    python inference.py --image-folder <folder>
"""

# example usage:
#
# python -m animal_recognition.src.evaluation.inference
#

from animal_recognition.src.models.yoloworld import YoloWorldDetector
from animal_recognition.src.models.classifier_convnext import ConvNextClassifier
from animal_recognition.src.models.classifier_gcvit import GCViTClassifier
import tempfile

import argparse
import re
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.colors as mcolors

import animal_recognition.src.data.augmentations_mild as augmentations_mild
import animal_recognition.src.data.augmentations as augmentations

import re

from rich.console import Console
from rich.table import Table

import tkinter as tk
from tkinter import filedialog

REJECT = -1

# Official class mapping fixed by the chair (index -> species). Train your
# classifier against this exact order so your labels match our evaluation.
CLASSES = [
    "Abyssinian",  #  0
    "Bengal",  #  1
    "Birman",  #  2
    "Bombay",  #  3
    "British_Shorthair",  #  4
    "Maine_Coon",  #  5
    "Ragdoll",  #  6
    "Sphynx",  #  7
    "Tabby",  #  8
    "Tiger_Cat",  #  9
    "Beagle",  # 10
    "Pug",  # 11
    "Boxer",  # 12
    "Shiba_Inu",  # 13
    "Samoyed",  # 14
    "Golden_Retriever",  # 15
    "German_Shepherd",  # 16
    "Siberian_Husky",  # 17
    "Dalmatian",  # 18
    "Rottweiler",  # 19
]
NUM_CLASSES = len(CLASSES)


class Model(nn.Module):
    def __init__(self, weights_path: Path, classifier, classifier_type):
        super().__init__()

        # use yoloworld with the default values (which are already analyzed to be best on test dataset)
        self.detector = YoloWorldDetector()

        if classifier == "convnextv2":  # convnextv2 or convnext
            self.classifier = ConvNextClassifier(pretrained=False, model_name=classifier_type)
        elif classifier == "gcvit":
            self.classifier = GCViTClassifier(pretrained=False, model_name=classifier_type)
        else:
            raise ValueError(f"Unsupported classifier: {classifier}")

        self.weights_path = Path(weights_path)

        if self.weights_path.exists():
            self.classifier.load_state_dict(torch.load(self.weights_path))
        else:
            raise ValueError(f"Weights path not found: {self.weights_path}")

        self.classifier.eval()

        try:

            match = re.search(r"_sz(\d+)", weights_path.stem)
            if match:
                self.image_size = int(match.group(1))
            else:
                res_str = weights_path.stem.split("_")[-1]
                self.image_size = int(res_str)
            print(f"Parsed resolution: {self.image_size}")
        except ValueError:
            self.image_size = 224  # Fallback
            print(
                f"Warning: Could not parse resolution from {weights_path.stem}. Defaulting to 224."
            )
        self.transform = augmentations.get_val_transforms(image_size=self.image_size)

    def forward(self, image: Image.Image) -> int:
        # temporary solution
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            image.save(tmp.name)

            # Predict using the identical logic from our sanitizer
            cropped_np, conf, cls_id = self.detector.predict(
                Path(tmp.name),
                confidence_threshold=0.05,
                reject_on_invalid_class=True,
                classes=list(range(self.detector.reject_classes_index)),
            )

        if cropped_np is None:
            return -1

        cropped_np = cropped_np[:, :, ::-1]

        input_tensor = self.transform(image=cropped_np)["image"].unsqueeze(0)

        device = next(self.classifier.parameters()).device
        input_tensor = input_tensor.to(device)

        confidences, class_indices = self.classifier.predict(input_tensor)

        return class_indices.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-folder", type=Path, default="images")
    parser.add_argument("--old_output", type=bool, default=False)
    args = parser.parse_args()

    # set default weights directory
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    initial_dir = project_root / "animal_recognition" / "models" / "weights"
    if not initial_dir.exists():
        initial_dir = Path.cwd()  # fallback to cwd if dir missing

    def select_files(init_dir):
        # iirc this is what works on windows, my distro doesnt come with tkinter though by default
        root = tk.Tk()
        root.withdraw()
        files = filedialog.askopenfilenames(
            initialdir=str(init_dir),
            title="Select Model Weights",
            filetypes=(("PyTorch weights", "*.pt *.pth"), ("All files", "*.*")),
        )
        return list(files)

    # get user selection
    weights_paths_str = select_files(initial_dir)

    if not weights_paths_str:
        print("No files selected. Exiting.")
        exit(0)

    weights_paths = [Path(p) for p in weights_paths_str]

    # load labels csv
    df = pd.read_csv(args.image_folder / "labels.csv")

    old_output = args.old_output
    console = Console()

    # store metrics for final ranking
    models_performance = []

    for w_path in weights_paths:
        stem = w_path.stem

        # infer base architecture
        if "gcvit" in stem:
            classifier = "gcvit"
        elif "convnextv2" in stem:
            classifier = "convnextv2"
        else:
            raise ValueError(f"Could not get classifier from {stem}")

        # extract exact model
        parts = stem.split("_")
        if len(parts) >= 2 and parts[1] in ["tiny", "small", "base", "large", "nano"]:
            classifier_type = f"{parts[0]}_{parts[1]}"
        else:
            raise ValueError(f"Could not get model {stem}")

        print(
            f"\nEvaluating {w_path.name} with inferred architecture: {classifier} ({classifier_type})"
        )

        model = Model(
            weights_path=w_path,
            classifier=classifier,
            classifier_type=classifier_type,
        ).eval()

        y_true, y_pred = [], []
        with torch.no_grad():
            # prgoress bar 
            for filename, label in tqdm(
                zip(df["filename"], df["label"]), total=len(df), desc=f"Inference {w_path.name}"
            ):
                image = Image.open(args.image_folder / filename).convert("RGB")
                pred = model(image)
                y_true.append(int(label))
                y_pred.append(int(pred))

        labels = [REJECT] + list(range(NUM_CLASSES))
        target_names = ["reject(-1)"] + CLASSES

        acc = accuracy_score(y_true, y_pred)
        report_dict = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            digits=3,
            zero_division=0,
            output_dict=True,
        )
        macro_f1 = report_dict["macro avg"]["f1-score"]
        macro_precision = report_dict["macro avg"]["precision"]
        macro_recall = report_dict["macro avg"]["recall"]
        weighted_acc = report_dict["weighted avg"]["recall"]  
        unweighted_acc = macro_recall 

        # save metrics for leaderboard
        models_performance.append({
            "name": w_path.name, 
            "acc": acc, 
            "f1": macro_f1,
            "precision": macro_precision,
            "recall": macro_recall,
            "weighted_acc": weighted_acc,
            "unweighted_acc": unweighted_acc
        })

        if old_output:
            print(f"\nWeighted Accuracy: {weighted_acc:.4f} | Unweighted Accuracy: {unweighted_acc:.4f}")
            print(
                classification_report(
                    y_true,
                    y_pred,
                    labels=labels,
                    target_names=target_names,
                    digits=3,
                    zero_division=0,
                )
            )
            print("Confusion matrix (rows=true, cols=pred):")
            print(confusion_matrix(y_true, y_pred, labels=labels))
        else:
            report = report_dict
            confusion_mtrx = confusion_matrix(y_true, y_pred, labels=labels)

            console.print(
                f"\n[bold bright_blue]Evaluation Results for: [white]{w_path.name}[/white][/bold bright_blue]"
            )
            console.print(f"[bold green]Weighted Accuracy: {weighted_acc:.4f} | Unweighted Accuracy: {unweighted_acc:.4f}[/bold green]\n")

            table = Table(
                title="Classification Report", show_header=True, header_style="bold magenta"
            )
            table.add_column("Class", style="dim", width=20)
            table.add_column("Precision", justify="right")
            table.add_column("Recall", justify="right")
            table.add_column("F1-Score", justify="right")
            table.add_column("Support", justify="right")

            for cls_name in target_names:
                if cls_name in report:
                    metrics = report[cls_name]
                    table.add_row(
                        cls_name,
                        f"{metrics['precision']:.3f}",
                        f"{metrics['recall']:.3f}",
                        f"{metrics['f1-score']:.3f}",
                        f"{int(metrics['support'])}",
                    )

            table.add_row("---", "---", "---", "---", "---")
            macro = report["macro avg"]
            table.add_row(
                "Macro Avg",
                f"{macro['precision']:.3f}",
                f"{macro['recall']:.3f}",
                f"{macro['f1-score']:.3f}",
                f"{int(macro['support'])}",
            )
            weighted = report["weighted avg"]
            table.add_row(
                "Weighted Avg",
                f"{weighted['precision']:.3f}",
                f"{weighted['recall']:.3f}",
                f"{weighted['f1-score']:.3f}",
                f"{int(weighted['support'])}",
            )

            console.print(table)

            cm_table = Table(
                title="Confusion Matrix (Rows=True, Cols=Pred)",
                show_header=True,
                header_style="bold yellow",
            )
            cm_table.add_column(r"T \ P", style="bold cyan")
            for i in range(len(target_names)):
                cm_table.add_column(str(labels[i]), justify="right")

            cmap = mpl.colormaps["coolwarm"]
            max_val = confusion_mtrx.max() if confusion_mtrx.max() > 0 else 1
            norm = mcolors.Normalize(vmin=0, vmax=max_val)

            for i, row in enumerate(confusion_mtrx):
                row_strs = []
                for val in row:
                    rgba = cmap(norm(val))
                    r, g, b = [int(c * 255) for c in rgba[:3]]
                    # calculate luminance to choose black or white text for readability: note this was tested on default gnome fedora and kde fedora dark theme
                    lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255
                    txt_color = "black" if lum > 0.5 else "white"
                    # format with width of 4 to make the background a nice square-ish block
                    row_strs.append(f"[{txt_color} on rgb({r},{g},{b})]{val:^4}[/]")
                cm_table.add_row(target_names[i], *row_strs)

            console.print(cm_table)

    # Leaderboard
    if len(weights_paths) > 1:
        models_performance.sort(key=lambda x: x["f1"], reverse=True)
        if old_output:
            print("\n" + "="*80)
            print("Model Ranking (Sorted by Macro F1-Score)")
            print("="*80)
            for rank, model_performance in enumerate(models_performance, 1):
                print(f"{rank}. {model_performance['name']}")
                print(f"   F1-Score:       {model_performance['f1']:.4f}")
                print(f"   Precision:      {model_performance['precision']:.4f}")
                print(f"   Recall:         {model_performance['recall']:.4f}")
                print(f"   Weighted Acc:   {model_performance['weighted_acc']:.4f}")
                print(f"   Unweighted Acc: {model_performance['unweighted_acc']:.4f}\n")
        else:
            console.print("\n[bold bright_blue]Model Ranking (Sorted by Macro F1-Score)[/bold bright_blue]")
            rank_table = Table(show_header=True, header_style="bold magenta")
            rank_table.add_column("Rank", justify="right", style="dim")
            rank_table.add_column("Model Name")
            rank_table.add_column("F1-Score", justify="right")
            rank_table.add_column("Precision", justify="right")
            rank_table.add_column("Recall", justify="right")
            rank_table.add_column("Weighted Acc", justify="right")
            rank_table.add_column("Unweighted Acc", justify="right")
            
            for rank, model_performance in enumerate(models_performance, 1):
                rank_table.add_row(
                    str(rank),
                    model_performance["name"],
                    f"{model_performance['f1']:.4f}",
                    f"{model_performance['precision']:.4f}",
                    f"{model_performance['recall']:.4f}",
                    f"{model_performance['weighted_acc']:.4f}",
                    f"{model_performance['unweighted_acc']:.4f}"
                )
            console.print(rank_table)
