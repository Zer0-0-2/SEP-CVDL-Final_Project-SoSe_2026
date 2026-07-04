import os
import shutil
from pathlib import Path

import cv2
import datetime
from ultralytics import YOLOWorld, settings
import logging

logging_path = Path(__file__).resolve().parent.parent.parent.parent / "logs"
logging_path.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    filename="logs/yoloworld.log",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

CAT_BREEDS = {
    "Abyssinian",
    "Bengal",
    "Birman",
    "Bombay",
    "British_Shorthair",
    "Maine_Coon",
    "Ragdoll",
    "Sphynx",
    "Tabby",
    "Tiger_Cat",
}
DOG_BREEDS = {
    "Beagle",
    "Boxer",
    "Dalmatian",
    "German_Shepherd",
    "Golden_Retriever",
    "Pug",
    "Rottweiler",
    "Samoyed",
    "Shiba_Inu",
    "Siberian_Husky",
}


MODEL_CLASSES = [
    # ACCEPT
    "cat",
    "dog",
    "domestic cat",
    "domestic dog",
    "sphynx cat",
    "bombay cat",
    "birman cat",
    ## REJECT
    "tiger",
    "wild tiger",
    "drawing",
    "painting",
    "big tiger",
    "predator",
]


cutoff = MODEL_CLASSES.index("wild tiger")


VALID_TARGETS = [i for i in range(cutoff)]
INVALID_TARGETS = [i for i in range(cutoff, len(MODEL_CLASSES))]


# takes the images from animal_recognition/data/raw and puts the bounding boxes of the images into
# animal_recognition/data/processed in the same folder structure. If there is multiple cats and dogs in the images
# it takes the biggest bounding box of a cat or dog depending on the species (thus images with a dog and a cat in for exmaple)
# r/abyssinian will just return the cat since we presume that's the image we care about for training.
# this is probably not perfect but it should work for now. Make sure that your animal_recognition/data/processed is empty before use


current_dir = Path(__file__).resolve().parent
animal_recog_dir = current_dir.parent.parent


def process_dataset(
    raw_dir: Path = animal_recog_dir / "data" / "raw",
    processed_dir: Path = animal_recog_dir / "data" / "processed_yoloworld",
    rejected_dir: Path = animal_recog_dir / "data" / "rejected_yoloworld",
    model_name: str = "yolov8x-worldv2.pt",
    model_classes: list[str] = MODEL_CLASSES,
    valid_targets: list[int] = VALID_TARGETS,
    invalid_targets: list[int] = INVALID_TARGETS,
    debug: bool = False,
    model_confidence_threshold: float = 0.25,
    test_provided_image_folder: bool = False,  # set to true if you want to test against the images provided by johannes and ming,
    # in that case ignore all the x_dir parameters
):

    time_start = datetime.datetime.now()

    model = YOLOWorld(model=animal_recog_dir / "models" / model_name, verbose=False)

    # prevent from downloading ViT file into weights/clip
    settings.update({"weights_dir": str(animal_recog_dir / "models" / "weights")})

    model.set_classes(model_classes)

    # debug
    if debug:
        print(raw_dir)
        print(processed_dir)
        print(rejected_dir)
        # REMOVES FOLDERS!!!
        shutil.rmtree(processed_dir, ignore_errors=True)
        shutil.rmtree(rejected_dir, ignore_errors=True)

    if not test_provided_image_folder:
        for breed in os.listdir(raw_dir):
            breed_dir = raw_dir / breed

            out_breed_dir = processed_dir / breed
            out_breed_dir.mkdir(parents=True, exist_ok=True)

            out_rejected_breed_dir = rejected_dir / breed
            out_rejected_breed_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Processing {breed}")

            placeholder(
                model=model,
                breed_dir=breed_dir,
                out_breed_dir=out_breed_dir,
                out_rejected_breed_dir=out_rejected_breed_dir,
                valid_targets=valid_targets,
                invalid_targets=invalid_targets,
                model_confidence_threshold=model_confidence_threshold,
                test_provided_image_folder=False,
            )

        time_end = datetime.datetime.now()
        print(f"Done. Operation took {time_end - time_start}")

    else:
        logger.info(f"Processing test folder")
        result = placeholder(
            model=model,
            breed_dir=Path("images"),
            out_breed_dir=animal_recog_dir / "data" / "processed_yoloworld_provided",
            out_rejected_breed_dir=animal_recog_dir / "data" / "rejected_yoloworld_provided",
            valid_targets=valid_targets,
            invalid_targets=invalid_targets,
            model_confidence_threshold=model_confidence_threshold,
            test_provided_image_folder=True,
        )
        return result


def placeholder(
    model: YOLOWorld,
    breed_dir: Path,
    out_breed_dir: Path,
    out_rejected_breed_dir: Path,
    valid_targets: list[int] = VALID_TARGETS,
    invalid_targets: list[int] = INVALID_TARGETS,
    model_confidence_threshold: float = 0.25,
    test_provided_image_folder: bool = False,
):
    results_dict: dict[str, str] = {}
    for img_name in os.listdir(breed_dir):
        img_path = breed_dir / img_name
        img = cv2.imread(str(img_path))
        if img is None:
            logger.error(f"Failed to load file: {img_path}")
            continue

        results = model(
            img, conf=model_confidence_threshold, verbose=False
        )  # returns a list as result, not single item, verbose = false to make console less cluttered

        largest_area = 0
        best_box = None
        best_cls_id = None

        best_box_confidence = 0

        # https://docs.ultralytics.com/tasks/detect#results-output
        is_animal_but_not_cat_or_dog = False

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id in valid_targets and cls_id not in invalid_targets:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = box.conf[0].item()
                    area = (x2 - x1) * (y2 - y1)

                    if area > largest_area:
                        largest_area = area
                        best_box = (int(x1), int(y1), int(x2), int(y2))
                        best_cls_id = cls_id
                        best_box_confidence = confidence

        if best_box is not None:
            x1, y1, x2, y2 = best_box
            cropped = img[y1:y2, x1:x2]
            new_img_name = f"{best_box_confidence:.2f}_{img_name}"
            out_path = out_breed_dir / new_img_name

            cv2.imwrite(str(out_path), cropped)

        else:
            cv2.imwrite(str(out_rejected_breed_dir / img_name), img)

            # In case no dog or cat is found in a dog or cat breed folder
            # which should happen pretty rarely, but i remember seeing a couple of images
            # while scraping that had no dog or cat in them (example one person with a dog bite)
            logger.info(f"No cat or dog found in image: {img_path}, saving to rejected folder")



if __name__ == "__main__":
    # x model for better resutls

    process_dataset(model_name="yolov8x-worldv2.pt", debug=True)
