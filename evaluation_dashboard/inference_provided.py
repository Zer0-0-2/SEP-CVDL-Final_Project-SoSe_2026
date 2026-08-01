import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import gc
from config import WEIGHTS_DIR, PROVIDED_DATASET_DIR, CLASSES, PROVIDED_CLASSES, evaluation_cache_provided, logger
import os
import re

import sys
sys.path.append("/app")

from animal_recognition.src.models.yoloworld import YoloWorldDetector
from animal_recognition.src.data.augmentations_stronger import get_val_transforms
from inference import load_model

def precompute_provided_all_models(models_to_compute, disk_cache, model_hashes):
    if not models_to_compute:
        return
        
    logger.info(f"Provided evaluation: {len(models_to_compute)} models to compute.")
    
    labels_csv = PROVIDED_DATASET_DIR / "labels.csv"
    if not labels_csv.exists():
        logger.error(f"Provided dataset labels not found at {labels_csv}")
        return
        
    df = pd.read_csv(labels_csv)
    
    # 1. Precompute all YOLO crops
    logger.info("Initializing YoloWorldDetector for precomputation...")
    try:
        detector = YoloWorldDetector()
    except Exception as e:
        logger.error(f"Failed to load YoloWorldDetector: {e}")
        return

    logger.info("Precomputing YOLO crops for all images...")
    precomputed_data = []
    total_imgs = len(df)
    
    for img_idx, (filename, label) in enumerate(zip(df["filename"], df["label"])):
        if img_idx > 0 and img_idx % 30 == 0:
            logger.info(f"  YOLO processed {img_idx}/{total_imgs} images ({(img_idx/total_imgs):.1%} complete)")
            
        img_path = PROVIDED_DATASET_DIR / filename
        try:
            cropped_np, conf, cls_id = detector.predict(
                img_path,
                confidence_threshold=0.05,
                reject_on_invalid_class=True,
                classes=list(range(detector.reject_classes_index)),
            )
            
            if cropped_np is None:
                precomputed_data.append({
                    "filename": filename,
                    "label": int(label),
                    "is_reject": True,
                    "cropped_rgb": None
                })
            else:
                cropped_rgb = cropped_np[:, :, ::-1]
                precomputed_data.append({
                    "filename": filename,
                    "label": int(label),
                    "is_reject": False,
                    "cropped_rgb": cropped_rgb
                })
        except Exception as e:
            logger.error(f"Error YOLO processing {filename}: {e}")
            precomputed_data.append({
                "filename": filename,
                "label": int(label),
                "is_reject": True,
                "cropped_rgb": None
            })
            
    del detector
    gc.collect()
    torch.cuda.empty_cache()
    
    # 2. Batch process the classifiers
    chunk_size = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for i in range(0, len(models_to_compute), chunk_size):
        chunk = models_to_compute[i:i+chunk_size]
        logger.info(f"Provided eval [{i+1} to {min(i+chunk_size, len(models_to_compute))}/{len(models_to_compute)}] Loading chunk of {len(chunk)} models into VRAM...")
        
        loaded_models = {}
        model_transforms = {}
        
        for w in chunk:
            w_path = WEIGHTS_DIR / w
            try:
                m, _ = load_model(w)
                loaded_models[w] = m
                
                match = re.search(r"_sz(\d+)", w_path.stem)
                if match:
                    image_size = int(match.group(1))
                else:
                    try:
                        res_str = w_path.stem.split("_")[-1]
                        image_size = int(res_str)
                    except ValueError:
                        image_size = 224
                model_transforms[w] = get_val_transforms(image_size=image_size)
                
            except Exception as e:
                logger.error(f"Failed to load classifier for {w}: {e}")
                
        if not loaded_models:
            continue
            
        all_preds = {w: [] for w in loaded_models}
        all_labels = []
        misclassifications = {w: [] for w in loaded_models}
        
        logger.info(f"  Evaluating chunk on provided dataset ({total_imgs} images)...")
        with torch.no_grad():
            for data in precomputed_data:
                label = data["label"]
                all_labels.append(label)
                
                for w, m in loaded_models.items():
                    if data["is_reject"]:
                        pred = -1
                    else:
                        tensor = model_transforms[w](image=data["cropped_rgb"])["image"].unsqueeze(0).to(device)
                        confidences, class_indices = m.predict(tensor)
                        pred = class_indices.item()
                        
                    all_preds[w].append(pred)
                    
                    if pred != label:
                        t_label_str = "reject(-1)" if label == -1 else CLASSES[label]
                        p_label_str = "reject(-1)" if pred == -1 else CLASSES[pred]
                        misclassifications[w].append({
                            "image_path": str(PROVIDED_DATASET_DIR / data["filename"]),
                            "true_label": t_label_str,
                            "pred_label": p_label_str,
                            "confidence": 1.0
                        })
                        
        logger.info("  Computing metrics...")
        y_true = np.array(all_labels)
        
        for w in loaded_models:
            y_pred = np.array(all_preds[w])
            
            acc = accuracy_score(y_true, y_pred)
            labels = [-1] + list(range(len(CLASSES)))
            
            precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
            macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='macro', zero_division=0)
            weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
            
            class_metrics = []
            for c_idx, cls_name in enumerate(PROVIDED_CLASSES):
                class_metrics.append({
                    "Breed": cls_name,
                    "Precision": round(float(precision[c_idx]), 4),
                    "Recall": round(float(recall[c_idx]), 4),
                    "F1-Score": round(float(f1[c_idx]), 4),
                    "Support": int(support[c_idx])
                })
                
            class_metrics.append({
                "Breed": "macro avg",
                "Precision": round(float(macro_prec), 4),
                "Recall": round(float(macro_rec), 4),
                "F1-Score": round(float(macro_f1), 4),
                "Support": int(np.sum(support))
            })
            class_metrics.append({
                "Breed": "weighted avg",
                "Precision": round(float(weighted_prec), 4),
                "Recall": round(float(weighted_rec), 4),
                "F1-Score": round(float(weighted_f1), 4),
                "Support": int(np.sum(support))
            })
            
            df_metrics = pd.DataFrame(class_metrics)
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            
            result = {
                "accuracy": float(acc),
                "macro_f1": float(macro_f1),
                "macro_precision": float(macro_prec),
                "macro_recall": float(macro_rec),
                "df_metrics": df_metrics,
                "misclassifications": misclassifications[w],
                "confusion_matrix": cm.tolist()
            }
            
            evaluation_cache_provided[w] = result
            disk_cache[w] = (model_hashes[w], result)
            
        del loaded_models
        gc.collect()
        torch.cuda.empty_cache()
