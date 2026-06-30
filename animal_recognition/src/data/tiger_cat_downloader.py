import os
from datasets import load_dataset

def download_tiger_cat_images():
    # Set up destination directory
    DESTINATION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "Tiger_Cat"))
    os.makedirs(DESTINATION_DIR, exist_ok=True)
    
    print(f"Saving Tiger_Cat images to: {DESTINATION_DIR}")
    
    # ImageNet label for Tiger Cat (n02123159) is 282
    TARGET_LABEL = 282
        
    try:
        dataset = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)
        
        count = 0
        for item in dataset:
            # Check if this image belongs to the Tiger Cat synset (label 282)
            if item['label'] == TARGET_LABEL:
                img = item['image']
                
                img_path = os.path.join(DESTINATION_DIR, f"tiger_cat_{count+1:03d}.jpg")
                
                # Convert to RGB in case the image is Grayscale or RGBA
                if img.mode != "RGB":
                    img = img.convert("RGB")
                    
                img.save(img_path)
                count += 1
                
                print(f"Downloaded {count}/200 images... (Keep waiting, it's streaming!)")
                    
                if count >= 200:
                    print("\nSuccessfully downloaded 200 Tiger_Cat images!")
                    break
                    
    except Exception as e:
        print(f"\nError accessing dataset: {e}")

if __name__ == "__main__":
    download_tiger_cat_images()
