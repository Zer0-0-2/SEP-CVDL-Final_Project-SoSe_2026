import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import gc
from config import WEIGHTS_DIR, DATASET_DIR, CLASSES, evaluation_cache, logger
from utils import format_model_name, get_model_list

from animal_recognition.src.models.classifier_convnext import ConvNextClassifier
from animal_recognition.src.models.classifier_gcvit import GCViTClassifier
from animal_recognition.src.data.dataset import AnimalDataset
from animal_recognition.src.data.augmentations import get_val_transforms
from animal_recognition.src.models.yoloworld import YoloWorldDetector
from pathlib import Path
import cv2

_YOLO_INSTANCE = None
def load_model(weights_name):
    weights_path = WEIGHTS_DIR / weights_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    state_dict = torch.load(weights_path, map_location=device)
    
    is_torchvision = any(k.startswith("model.features.") for k in state_dict.keys())
    
    if is_torchvision:
        import torchvision.models as tv_models
        from torch import nn
        
        if "convnext_tiny" in weights_name:
            model = tv_models.convnext_tiny()
            model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(CLASSES))
        else:
            raise ValueError(f"Unknown torchvision model architecture for {weights_name}")
            
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
                
        model.load_state_dict(new_state_dict)
        model = model.to(device)
        model.eval()
        
        class TorchvisionWrapper:
            def __init__(self, m):
                self.model = m
            def predict(self, images):
                out = self.model(images)
                conf = torch.nn.functional.softmax(out, dim=1).max(dim=1).values
                cls = out.argmax(dim=1)
                return conf, cls
        return TorchvisionWrapper(model), device

    if "gcvit_tiny" in weights_name.lower():
        model = GCViTClassifier(model_name="gcvit_tiny")
    elif "convnextv2_tiny" in weights_name.lower():
        model = ConvNextClassifier(model_name="convnextv2_tiny")
    elif "convnextv2_base" in weights_name.lower():
        model = ConvNextClassifier(model_name="convnextv2_base")
    elif "gcvit_base" in weights_name.lower():
        model = GCViTClassifier(model_name="gcvit_base")
    elif "convnext_tiny_baseline" in weights_name.lower():
        model = ConvNextClassifier(model_name="convnext_tiny")
    else:
        raise ValueError(f"Unknown model: {weights_name}")
        
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, device

def evaluate_model(weights_name):
    print(f"Evaluating {weights_name}...")
    model, device = load_model(weights_name)
    val_transforms = get_val_transforms(image_size=224)
    
    dataset = AnimalDataset(DATASET_DIR, transform=val_transforms)
    print(f"Dataset: {len(dataset)} images, {len(dataset.classes)} classes")
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    all_preds = []
    all_labels = []
    all_confidences = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            if batch_idx > 0 and batch_idx % 20 == 0:
                print(f"Processed batch {batch_idx}/{len(loader)}")
            
            images = images.to(device)
            conf, cls = model.predict(images)
            
            all_preds.extend(cls.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_confidences.extend(conf.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    
    class_metrics = []
    for i, cls_name in enumerate(CLASSES):
        class_metrics.append({
            "Breed": cls_name,
            "Precision": round(float(precision[i]), 4),
            "Recall": round(float(recall[i]), 4),
            "F1-Score": round(float(f1[i]), 4),
            "Support": int(support[i])
        })
        
    df_metrics = pd.DataFrame(class_metrics)
    
    misclassifications = []
    for idx in range(len(all_labels)):
        if all_labels[idx] != all_preds[idx]:
            img_path = str(dataset.samples[idx][0])
            misclassifications.append({
                "image_path": img_path,
                "true_label": CLASSES[all_labels[idx]],
                "pred_label": CLASSES[all_preds[idx]],
                "confidence": float(all_confidences[idx])
            })
            
    result = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "df_metrics": df_metrics,
        "misclassifications": misclassifications
    }
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return result

def precompute_all_models_batched(models_to_compute, disk_cache, model_hashes, chunk_size=3):
    if not models_to_compute:
        return
        
    val_transforms = get_val_transforms(image_size=224)
    dataset = AnimalDataset(DATASET_DIR, transform=val_transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for i in range(0, len(models_to_compute), chunk_size):
        chunk = models_to_compute[i:i+chunk_size]
        logger.info(f"[{i+1} to {min(i+chunk_size, len(models_to_compute))}/{len(models_to_compute)}] Loading chunk of {len(chunk)} models into VRAM...")
        
        loaded_models = {}
        for w in chunk:
            m, _ = load_model(w)
            loaded_models[w] = m
            
        all_preds = {w: [] for w in chunk}
        all_confidences = {w: [] for w in chunk}
        all_labels = []
        
        logger.info(f"Evaluating chunk on dataset ({len(loader)} batches)") 
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(loader):
                if batch_idx > 0 and batch_idx % 16 == 0:
                    logger.info(f"  Processed batch {batch_idx}/{len(loader)} ({(batch_idx/len(loader)):.1%} complete)")
                all_labels.extend(labels.numpy())
                images = images.to(device)
                
                for w, m in loaded_models.items():
                    conf, cls = m.predict(images)
                    all_preds[w].extend(cls.cpu().numpy())
                    all_confidences[w].extend(conf.cpu().numpy())
                    
        logger.info("Computing metrics for evaluated models...")
        all_labels_np = np.array(all_labels)
        for w in chunk:
            logger.info(f"  Computing metrics for: {w}")
            preds = np.array(all_preds[w])
            confs = np.array(all_confidences[w])
            
            acc = accuracy_score(all_labels_np, preds)
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels_np, preds, average=None, zero_division=0)
            macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(all_labels_np, preds, average='macro', zero_division=0)
            
            class_metrics = []
            for c_idx, cls_name in enumerate(CLASSES):
                class_metrics.append({
                    "Breed": cls_name,
                    "Precision": round(float(precision[c_idx]), 4),
                    "Recall": round(float(recall[c_idx]), 4),
                    "F1-Score": round(float(f1[c_idx]), 4)
                })
            df_metrics = pd.DataFrame(class_metrics)
            
            misclassifications = []
            for idx in range(len(all_labels_np)):
                if all_labels_np[idx] != preds[idx]:
                    img_path = str(dataset.samples[idx][0])
                    misclassifications.append({
                        "image_path": img_path,
                        "true_label": CLASSES[all_labels_np[idx]],
                        "pred_label": CLASSES[preds[idx]],
                        "confidence": float(confs[idx])
                    })
                    
            cm = confusion_matrix(all_labels_np, preds, labels=range(len(CLASSES)))
                    
            result = {
                "accuracy": float(acc),
                "macro_f1": float(macro_f1),
                "macro_precision": float(macro_prec),
                "macro_recall": float(macro_rec),
                "df_metrics": df_metrics,
                "misclassifications": misclassifications,
                "confusion_matrix": cm.tolist()
            }
            evaluation_cache[w] = result
            disk_cache[w] = (model_hashes[w], result)
            
        del loaded_models
        gc.collect()
        torch.cuda.empty_cache()

def analyze_image_all_models(image, experiment="All models", show_filenames=False):
    global _YOLO_INSTANCE
    logger.info(f"Custom Upload -> Running analysis across models in stage: {experiment}")
    
    if _YOLO_INSTANCE is None:
        _YOLO_INSTANCE = YoloWorldDetector()
        
    cropped_np, conf, cls_id = _YOLO_INSTANCE.predict(
        Path(image),
        confidence_threshold=0.05,
        reject_on_invalid_class=False,
        classes=list(range(_YOLO_INSTANCE.reject_classes_index)),
    )
    
    if cropped_np is not None:
        cropped_rgb = cropped_np[:, :, ::-1].copy()
    else:
        img = cv2.imread(image)
        cropped_rgb = img[:, :, ::-1].copy()
        
    val_transforms = get_val_transforms(image_size=224)
    img_tensor = val_transforms(image=cropped_rgb)["image"].unsqueeze(0)
    models = get_model_list(experiment)
    
    results = []
    
    for weights_name in models:
        logger.debug(f"Predicting with model: {weights_name}")
        model, device = load_model(weights_name)
        img_tensor = img_tensor.to(device)
        
        with torch.no_grad():
            conf, cls = model.predict(img_tensor)
            conf = float(conf.cpu())
            pred_idx = int(cls.cpu())
            pred_class = CLASSES[pred_idx]
            
            results.append({
                "Model": weights_name if show_filenames else format_model_name(weights_name),
                "Prediction": pred_class,
                "Confidence": f"{conf:.2%}"
            })
            
        del model
        gc.collect()
        torch.cuda.empty_cache()
            
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by="Confidence", ascending=False)
        majority_class = df["Prediction"].mode()[0]
        vote_count = (df["Prediction"] == majority_class).sum()
        total_votes = len(df)
        majority_str = f"### 🗳️ Majority Vote: **{majority_class}** ({vote_count}/{total_votes} models)"
    else:
        majority_str = "### 🗳️ Majority Vote: Waiting for inference..."
        
    return df, majority_str, cropped_rgb
