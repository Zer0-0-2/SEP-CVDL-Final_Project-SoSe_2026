"""Compares classifier accuracy with vs. without the OOD gate on one trained
model, and saves the comparison as a bar chart under ./oodGate_stats/.

Reuses InferenceBackup.Model (same Detector -> Classifier pipeline, same
weight loading / architecture handling) so this script and InferenceBackup.py
can't drift out of sync. For each image it takes the logits from
Model.get_logits() once and derives two predictions from them:
  - no gate:   plain argmax (detector reject -1 still applies if no crop found)
  - with gate: OODGate(mode) decides accept/reject on top of the argmax

    source venv/bin/activate
    python -m animal_recognition.src.evaluation.ood_gate_accuracy_comparison \
        --image-folder animal_recognition/data/eval_flat \
        --classifier-type convnext \
        --model-name convnext_small \
        --weights-path animal_recognition/src/models/models_weights/convnext_small_aug_val_warmup_preFalse_bs32_lr0.003_wd0.05_ls0.2_sz224_augvetted.pt \
        --ood-gate softmax_threshold \
        --ood-threshold 0.5

options:
Dateiname-Präfix	--classifier-type	--model-name
convnext_tiny_*	    convnext	        convnext_tiny
convnext_small_*	convnext	        convnext_small
gcvit_tiny_*	    gcvit	            gcvit_tiny
convnextv2_tiny_*	—	                funktioniert aktuell nicht
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from animal_recognition.src.evaluation.InferenceBackup import Model
from animal_recognition.src.config import load_config

REJECT = -1
OUTPUT_DIR = Path("oodGate_stats")


def predict_both(model: Model, image: Image.Image) -> tuple[int, int, torch.Tensor]:
    """Returns (pred_no_gate, pred_with_gate, logits) for a single image, from the
    same logits."""
    logits = model.get_logits(image)
    if logits is None:
        return REJECT, REJECT, None
    pred_no_gate = logits.squeeze(0).argmax().item()
    pred_with_gate = model.ood_gate(logits)
    return pred_no_gate, pred_with_gate, logits


if __name__ == "__main__":
    cfg = load_config()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-folder", type=Path, required=True)
    parser.add_argument("--weights-path", type=Path, required=True)
    parser.add_argument("--classifier-type", type=str, default="convnext", choices=["convnext", "gcvit"])
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=(
            "Backbone variant, must match the checkpoint's architecture "
            "(encoded in its filename). Defaults to 'convnext_tiny' "
            "(--classifier-type convnext) or 'gcvit_tiny' (--classifier-type gcvit) "
            "if omitted."
        ),
    )
    parser.add_argument("--ood-gate", type=str, default=cfg.pipeline.ood_gate,
                         choices=["softmax_threshold", "energy", "temperature_scaling"])
    parser.add_argument("--ood-threshold", type=float, default=cfg.ood.threshold)
    parser.add_argument("--ood-temperature", type=float, default=cfg.ood.temperature)
    args = parser.parse_args()

    df = pd.read_csv(args.image_folder / "labels.csv")
    model = Model(
        weights_path=args.weights_path,
        classifier_type=args.classifier_type,
        model_name=args.model_name,
        ood_gate_mode=args.ood_gate,
        ood_threshold=args.ood_threshold,
        ood_temperature=args.ood_temperature,
    ).eval()

    y_true, y_pred_no_gate, y_pred_with_gate = [], [], []
    cases = []  # Track all cases for later analysis
    with torch.no_grad():
        for filename, label in tqdm(zip(df["filename"], df["label"]), total=len(df)):
            image = Image.open(args.image_folder / filename).convert("RGB")
            pred_no_gate, pred_with_gate, logits = predict_both(model, image)
            y_true.append(int(label))
            y_pred_no_gate.append(int(pred_no_gate))
            y_pred_with_gate.append(int(pred_with_gate))
            
            logits_str = ""
            confidence_str = ""
            if logits is not None:
                logits_str = " ".join(f"{v:.3f}" for v in logits.squeeze(0).cpu().tolist()[:10])
                # compute a gate-specific confidence / score for display
                if args.ood_gate == "softmax_threshold":
                    probs = torch.softmax(logits, dim=1)
                    conf = probs.max().item()
                    confidence_str = f"softmax={conf:.3f}"
                elif args.ood_gate == "temperature_scaling":
                    T = float(args.ood_temperature)
                    probs = torch.softmax(logits / T, dim=1)
                    conf = probs.max().item()
                    confidence_str = f"softmax_T={conf:.3f}"
                elif args.ood_gate == "energy":
                    T = float(args.ood_temperature)
                    energy = -T * torch.logsumexp(logits / T, dim=1)
                    conf = energy.item()
                    confidence_str = f"energy={conf:.3f}"

            gate_status = "✓ Accepted" if pred_with_gate != REJECT else "✗ Rejected"

            cases.append({
                "filename": filename,
                "true_label": int(label),
                "pred_no_gate": int(pred_no_gate),
                "pred_with_gate": int(pred_with_gate),
                "logits_str": logits_str,
                "gate_status": gate_status,
                "confidence": confidence_str,
            })

    acc_no_gate = accuracy_score(y_true, y_pred_no_gate)
    acc_with_gate = accuracy_score(y_true, y_pred_with_gate)

    print(f"\nAccuracy without OOD gate: {acc_no_gate:.4f}")
    print(f"Accuracy with OOD gate ({args.ood_gate}): {acc_with_gate:.4f}")
    
    # Print all test cases
    print("\n" + "="*120)
    print("DETAILED PREDICTIONS FOR ALL TEST CASES:")
    print("="*120)
    for case in cases:
        print(
            f"[{case['filename']}] True: {case['true_label']:2d} | "
            f"Logits: [{case['logits_str']}...] | "
            f"No Gate: {case['pred_no_gate']:2d} → With Gate: {case['pred_with_gate']:2d} {case['gate_status']}"
        )
    
    # Identify and display cases where OOD gate changed the prediction
    print("\n" + "="*100)
    print("CASES WHERE OOD GATE REJECTED (CONFOUNDER DETECTION):")
    print("="*100)
    
    rejected_cases = [c for c in cases if c["pred_with_gate"] == REJECT]
    unchanged_rejections = [c for c in rejected_cases if c["pred_no_gate"] == REJECT]
    gate_rejections = [c for c in rejected_cases if c["pred_no_gate"] != REJECT]
    
    if not rejected_cases:
        print("No cases where OOD gate rejected predictions.")
    else:
        correct_rejects = 0  # True negatives - rejected confounders/wrong predictions
        incorrect_rejects = 0  # False negatives - rejected correct predictions
        
        for case in gate_rejections:
            was_correct = case["pred_no_gate"] == case["true_label"]
            if was_correct:
                incorrect_rejects += 1
                status = "✗ FALSE NEGATIVE - Rejected a correct prediction"
            else:
                correct_rejects += 1
                status = "✓ TRUE NEGATIVE - Correctly rejected a wrong prediction"
            
            print(
                f"  {case['filename']}\n"
                f"    True Label: {case['true_label']}\n"
                f"    Prediction (no gate): {case['pred_no_gate']:2d}  →  Rejected by gate: -1  [{status}]\n"
            )
        
        if unchanged_rejections:
            print(f"  Note: {len(unchanged_rejections)} cases were already rejected before the OOD gate and are excluded from these counts.")
        
        print(f"\n  Summary: {correct_rejects} correct rejections (avoided false positives), "
              f"{incorrect_rejects} incorrect rejections (false negatives)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    bars = ax.bar(
        ["without OOD-Gate", f"with OOD-Gate\n({args.ood_gate})"],
        [acc_no_gate, acc_with_gate],
        color=["#888888", "#2b7de9"],
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{args.classifier_type} — {args.weights_path.name}")
    for bar, acc in zip(bars, [acc_no_gate, acc_with_gate]):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.02, f"{acc:.3f}", ha="center")

    # show OOD threshold on the plot
    ax.text(
        0.98,
        0.02,
        f"OOD threshold: {args.ood_threshold}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#333333",
    )
    fig.tight_layout()
    out_path = OUTPUT_DIR / f"ood_gate_accuracy_{args.classifier_type}_{args.ood_gate}.jpg"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
