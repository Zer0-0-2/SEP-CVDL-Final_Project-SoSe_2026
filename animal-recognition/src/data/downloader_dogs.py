from datasets import load_dataset
from datasets import concatenate_datasets
from huggingface_hub import hf_hub_download
from datasets import Image
import zipfile
import os


# 1. Datensätze laden (ohne Barkopedia, da dies ein Audio-Datensatz ist)
print("Lade Datensätze herunter...")
ds_saugat = load_dataset("Saugatkafley/dog-breed-classification", split="train")
ds_lpastor = load_dataset("lpastor75/dog-breed-classification", split="train")



print("Lade Bilder für lpastor herunter...")
zip_path = hf_hub_download(repo_id="lpastor75/dog-breed-classification", filename="dog-images.zip", repo_type="dataset")

extract_dir = os.path.abspath("data/lpastor_extracted")


if not os.path.exists(extract_dir):
    print(f"Entpacke Bilder nach {extract_dir}...")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

def add_image_path(example):
    example["image"] = os.path.join(extract_dir, example["path"])
    return example

ds_lpastor = ds_lpastor.map(add_image_path)
ds_lpastor = ds_lpastor.cast_column("image", Image())

def normalize_label(example, label_column_name="label"):
    # Label in Text umwandeln und kleinschreiben
    raw_label = str(example[label_column_name]).lower()
    
        
    # Unterstriche durch Leerzeichen ersetzen und Leerzeichen bereinigen
    raw_label = raw_label.replace("_", " ").strip()
    
    # Neues, bereinigtes Label als Text speichern
    example["unified_label_text"] = raw_label
    return example

# 3. Bereinigung auf alle Datensätze anwenden
# Hinweis: Die Spalte, die das Label enthält, heißt je nach Ersteller manchmal anders (z.B. 'label', 'breed' oder 'text'). 
# Passe den Parameter 'label_column_name' an, falls das Skript eine Spalte nicht findet.
print("Vereinheitliche Labels...")
ds_saugat = ds_saugat.map(lambda x: normalize_label(x, label_column_name="label"))
ds_lpastor = ds_lpastor.map(lambda x: normalize_label(x, label_column_name="label"))

# 4. Alle einzigartigen Hunde-Rassen aus allen Datensätzen sammeln
all_labels = set()
for ds in [ds_saugat, ds_lpastor]:
    all_labels.update(ds["unified_label_text"])

# 5. Globales Wörterbuch für die neuen, einheitlichen IDs erstellen (0 bis N)
label2id = {label: idx for idx, label in enumerate(sorted(all_labels))}
id2label = {idx: label for label, idx in label2id.items()}
print(f"Insgesamt {len(label2id)} einzigartige Hunderassen über alle Datensätze gefunden.")

# 6. Funktion zum Zuweisen der finalen Integer-ID
def assign_unified_id(example):
    example["unified_label_id"] = label2id[example["unified_label_text"]]
    return example

# IDs zuweisen
ds_saugat = ds_saugat.map(assign_unified_id)
ds_lpastor = ds_lpastor.map(assign_unified_id)

# 7. (Optional) Nur die relevanten Spalten behalten und Datensätze zusammenfügen
def filter_columns(ds):
    return ds.select_columns(["image", "unified_label_id", "unified_label_text"])

combined_dataset = concatenate_datasets([
    filter_columns(ds_saugat),
    filter_columns(ds_lpastor),
])

print(f"Fertig! Der kombinierte Datensatz enthält {len(combined_dataset)} Bilder.")

combined_dataset.save_to_disk("data/processed/hf_dog_breeds")