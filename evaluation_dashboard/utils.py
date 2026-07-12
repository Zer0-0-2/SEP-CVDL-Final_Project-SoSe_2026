import hashlib
from config import WEIGHTS_DIR

def compute_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_model_list():
    if not WEIGHTS_DIR.exists():
        return []
    weights = list(WEIGHTS_DIR.glob("*.pt"))
    return [w.name for w in weights]

def format_model_name(filename):
    name = filename.replace(".pt", "")
    parts = name.split("_")
    
    arch = "Unknown"
    if "convnextv2_tiny" in name: arch = "ConvNeXtV2 Tiny"
    elif "convnext_tiny" in name: arch = "ConvNeXt Tiny"
    elif "gcvit_tiny" in name: arch = "GCViT Tiny"
    elif "convnextv2_base" in name: arch = "ConvNextV2 Base"
    elif "gcvit_base" in name: arch = "GCViT Base"
    
    method = ""
    if "bitfit" in name: method = "BitFit"
    elif "conservative_finetune" in name: method = "Conservative Finetune"
    elif "aggressive_finetune" in name: method = "Aggressive Finetune"
    elif "layer_decay" in name: method = "Layer Decay"
    elif "linear_probe" in name: method = "Linear Probe"
    elif "partial_freeze" in name: method = "Partial Freeze"
    elif "custom_wd" in name: method = "Custom WD"
    elif "paper_rep" in name: method = "Paper Reproduction"
    elif "baseline" in name: method = "Baseline"
    elif "cosinelr_with_warmup_optimized_weight_decay" in name: method = "Optimized WD"
    elif "cosinelr_with_warmup" in name: method = "Cosine Warmup"
    
    lr = ""
    for p in parts:
        if p.startswith("lr"):
            lr = p[2:]
            break
            
    sched = ""
    if "Cosine" in name: sched = "CosineLR"
    elif "Step" in name: sched = "StepLR"
        
    pretrained = ""
    if "preTrue" in name: pretrained = "Pretrained: True"
    elif "preFalse" in name: pretrained = "Pretrained: False"
    
    parts_to_join = []
    if method: parts_to_join.append(method)
    if pretrained: parts_to_join.append(pretrained)
    if lr: parts_to_join.append(f"LR: {lr}")
    if sched: parts_to_join.append(sched)
    
    details = " | ".join(parts_to_join)
    if details:
        return f"{arch} ({details})"
    return arch
