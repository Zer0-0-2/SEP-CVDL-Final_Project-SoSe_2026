import logging
from pathlib import Path
import animal_recognition.src.models.yoloworld as yoloworld
import os
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

DEFAULT_SCRAPED_IMAGES_FOLDER = (
    PROJECT_ROOT / "animal_recognition" / "data" / "raw"
)  # NOTE: this contains subfolderse with the actual images.
# Make sure that this does not actually contain any files

DEFAULT_MODEL_CLASSES = [
    "cat",
    "dog",
    "domestic cat",
    "domestic dog",
    "sphynx cat",
    "bombay cat",
    "birman cat",  # 6
    "tiger",
    "big tiger",
    "fox",
    "wolf",
    "coyote",
    "drawing",
    "painting",
]
DEFAULT_REJECT_CLASSES_INDEX = DEFAULT_MODEL_CLASSES.index("tiger")

DEFAULT_CONFIDENCE_THRESHOLD = 0.05


def main(
    image_folder: Path = DEFAULT_SCRAPED_IMAGES_FOLDER,
    model_name: str = "yolov8x-worldv2.pt",
    model_classes: list[str] = DEFAULT_MODEL_CLASSES,
    reject_classes_index: int = DEFAULT_REJECT_CLASSES_INDEX,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
):
    accepted = image_folder / ".." / "processed" / "accepted"
    rejected = image_folder / ".." / "processed" / "rejected"

    if not accepted.exists():
        accepted.mkdir(parents=True, exist_ok=True)
    if not rejected.exists():
        rejected.mkdir(parents=True, exist_ok=True)

    model = yoloworld.YoloWorldDetector(
        model_name=model_name,
        model_classes=model_classes,
        reject_classes_index=reject_classes_index,
    )

    for folder in image_folder.iterdir():
        if folder.name in ("accepted", "rejected") or not folder.is_dir():
            continue
            
        accepted_folder = accepted / folder.name
        rejected_folder = rejected / folder.name
        accepted_folder.mkdir(parents=True, exist_ok=True)
        rejected_folder.mkdir(parents=True, exist_ok=True)

        for image_file in folder.iterdir():
            logging.info(f"Processing {image_file.name} in folder {folder.name}")
            if not image_file.is_file():
                continue
            cropped, confidence, class_id = model.predict(
                image_file, 
                confidence_threshold=confidence_threshold, 
                reject_on_invalid_class=True, 
            )
            if cropped is None:
                logging.info(
                    f"Rejected {image_file.name} with confidence {confidence} and class_id {class_id}"
                )
                cv2.imwrite(str(rejected_folder / image_file.name), cv2.imread(str(image_file)))
            elif cropped is not None:
                logging.info(
                    f"Accepted {image_file.name} with confidence {confidence} and class_id {class_id}"
                )
                cv2.imwrite(str(accepted_folder / image_file.name), cropped)
if __name__ == "__main__":
    logging_path = PROJECT_ROOT / "logs"

    logging_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        format="{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
        filename="logs/sanitizer.log",
        level=logging.INFO,
    )

    logger = logging.getLogger(__name__)
    # print(PROJECT_ROOT)
    main()