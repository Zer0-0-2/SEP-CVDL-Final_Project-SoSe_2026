import os
import cv2
from pathlib import Path
from ultralytics import YOLO
import shutil

'''
            CLASSES = [
                "Abyssinian",         #  0
                "Bengal",             #  1
                "Birman",             #  2
                "Bombay",             #  3
                "British_Shorthair",  #  4
                "Maine_Coon",         #  5
                "Ragdoll",            #  6
                "Sphynx",             #  7
                "Tabby",              #  8
                "Tiger_Cat",          #  9
                "Beagle",             # 10 DOGS
                "Pug",                # 11
                "Boxer",              # 12
                "Shiba_Inu",          # 13
                "Samoyed",            # 14
                "Golden_Retriever",   # 15
                "German_Shepherd",    # 16
                "Siberian_Husky",     # 17
                "Dalmatian",          # 18
                "Rottweiler",         # 19
            ]
            
''' 
CAT_BREEDS = {
    'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair',
    'Maine_Coon', 'Ragdoll', 'Sphynx', 'Tabby', 'Tiger_Cat'
}
DOG_BREEDS = {
    'Beagle', 'Boxer', 'Dalmatian', 'German_Shepherd', 'Golden_Retriever',
    'Pug', 'Rottweiler', 'Samoyed', 'Shiba_Inu', 'Siberian_Husky'
}
# takes the images from animal_recognition/data/raw and puts the bounding boxes of the images into 
# animal_recognition/data/processed in the same folder structure. If there is multiple cats and dogs in the images
# it takes the biggest bounding box of a cat or dog depending on the species (thus images with a dog and a cat in for exmaple) 
# r/abyssinian will just return the cat since we presume that's the image we care about for training.
# this is probably not perfect but it should work for now. Make sure that your animal_recognition/data/processed is empty before use
def process_dataset(raw_dir: Path, processed_dir: Path, rejected_dir: Path, model_name: str, debug: bool = False):
    model = YOLO(model = animal_recog_dir / 'models' / model_name, verbose = False)
    # debug 
    if debug:
        print(raw_dir) 
        print(processed_dir)
        print(rejected_dir)
        shutil.rmtree(processed_dir, ignore_errors=True)
        shutil.rmtree(rejected_dir, ignore_errors=True)
        
    for breed in os.listdir(raw_dir):
        breed_dir = raw_dir / breed
        
            
        if breed in CAT_BREEDS:
            target_class = 15  
        elif breed in DOG_BREEDS:
            target_class = 16
        else:
            print(f"error, please download the correct dataset from filen and unpack it correctly -_- for breed: {breed}")
            exit(1)
            
            
        out_breed_dir = processed_dir / breed
        out_breed_dir.mkdir(parents=True, exist_ok=True)
        
        out_rejected_breed_dir = rejected_dir / breed
        out_rejected_breed_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing {breed}")
        
        for img_name in os.listdir(breed_dir):
                
            img_path = breed_dir / img_name
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue
                
            results = model(img, verbose=False) # returns a list as result, not single item, verbose = false to make console less cluttered
            
            largest_area = 0
            best_box = None
            
            
            # https://docs.ultralytics.com/tasks/detect#results-output 
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    if cls_id == target_class:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        confidence = box.conf[0].item()
                        area = (x2 - x1) * (y2 - y1)
                        if area > largest_area:
                            largest_area = area
                            best_box = (int(x1), int(y1), int(x2), int(y2))
                            best_box_confidence = confidence
                            img_name = str(best_box_confidence) + "_" + img_name 
            
            if best_box is not None:
                x1, y1, x2, y2 = best_box
                cropped = img[y1:y2, x1:x2]
            else:
                cv2.imwrite(str(out_rejected_breed_dir / img_name), img) 
                print(f"No cat or dog found in image: {img_path}, saving to rejected folder") 
                continue # In case no dog or cat is found in a dog or cat breed folder
                            # which should happen pretty rarely, but i remember seeing a couple of images
                            # while scraping that had no dog or cat in them (example one person with a dog bite)
                
            out_path = out_breed_dir / img_name
            cv2.imwrite(str(out_path), cropped)

if __name__ == '__main__':
    current_dir = Path(__file__).resolve().parent
    animal_recog_dir = current_dir.parent.parent
    raw_dir = animal_recog_dir / 'data' / 'raw'
    processed_dir = animal_recog_dir / 'data' / 'processed'
    rejected_dir = animal_recog_dir / 'data' / 'rejected'
    
    # x model for better resutls 
    process_dataset(raw_dir, processed_dir, rejected_dir, "yolo26x", debug=True)
    

