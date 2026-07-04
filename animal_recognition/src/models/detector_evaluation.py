## this is a random file to evaluate the detectors
# run it using
# python -m animal_recognition.src.models.detector_evaluation


from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

import animal_recognition.src.models.yoloworld as yoloworld


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_IMAGE_FOLDER = PROJECT_ROOT / "images"
DEFAULT_LABELS_FILE = DEFAULT_IMAGE_FOLDER / "labels.csv"


CONFIDENCE_THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
MODELS = ["YOLOv8s-worldv2", "YOLOv8m-worldv2", "YOLOv8l-worldv2", "YOLOv8x-worldv2"]

CAT_OR_DOG_LABELS = [i for i in range(0, 20)]
CONFOUNDER_LABEL = -1


def evaluate_detector(
    confidence_threshold: float,
    classes: list[str],
    valid_targets: list[int],
    invalid_targets: list[int],
) -> dict[str, str]:
    result = yoloworld.process_dataset(
        model_classes=classes,
        valid_targets=valid_targets,
        invalid_targets=invalid_targets,
        debug=False,
        model_confidence_threshold=confidence_threshold,
        test_provided_image_folder=True,
    )
    return result


def evaluate_image_folder(
    out_image_dir: Path,
    out_csv_path: Path,
    image_folder: Path = DEFAULT_IMAGE_FOLDER,
):

    classes, valid_targets, invalid_targets = compute_classes()

    labels_path = image_folder / "labels.csv"
    df = pd.read_csv(labels_path)

    for single_class in classes:
        for confidence_threshold in CONFIDENCE_THRESHOLDS:
            logging.info(
                f"Evaluating detector with confidence threshold {confidence_threshold} and class set {single_class}"
            )
            image_label_dict = evaluate_detector(
                confidence_threshold, single_class, valid_targets, invalid_targets
            )
        y_true = []
        y_pred = []
        for filename, label in df[["filename", "label"]].itertuples(index=False):
            y_true = label
            y_pred = image_label_dict[filename]
            
            if y_true != -1 and y_pred == "notcatordog"

            correct = sum(expected == predicted for expected, predicted in zip(y_true, y_pred))
            total = len(y_true)
            accuracy = correct / total

    return accuracy


def main():

    logging_path = PROJECT_ROOT / "logs"
    logging_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        format="{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
        filename=str(logging_path / "detector_eval.log"),
        level=logging.INFO,
    )

    evaluate_image_folder(
        image_folder=DEFAULT_IMAGE_FOLDER,
        out_image_dir=PROJECT_ROOT / "animal_recognition" / "data",
        out_csv_path=PROJECT_ROOT / "animal_recognition" / "data" / "evaluation_results.csv",
    )


def compute_classes():
    classes = []
    valid_targets = []
    invalid_targets = []

    # All options
    classes1 = [
        "animal",
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
    cutoff1 = classes1.index("tiger")
    valid_targets1 = [i for i in range(1, cutoff1)]
    invalid_targets1 = [i for i in range(cutoff1, len(classes1))]

    classes.append(classes1)
    valid_targets.append(valid_targets1)
    invalid_targets.append(invalid_targets1)

    # Just cat and dog
    classes2 = [
        "animal",
        # ACCEPT
        "cat",
        "dog",
    ]
    valid_targets2 = [1, 2]
    invalid_targets2 = []

    classes.append(classes2)
    valid_targets.append(valid_targets2)
    invalid_targets.append(invalid_targets2)

    # some more options but w/o duplicate reject classes
    classes3 = [
        "animal",
        # ACCEPT
        "cat",
        "dog",
        "domestic cat",
        "domestic dog",
        ## REJECT
        "tiger",
        "drawing",
    ]
    cutoff3 = classes3.index("tiger")
    valid_targets3 = [i for i in range(1, cutoff3)]
    invalid_targets3 = [i for i in range(cutoff3, len(classes3))]

    classes.append(classes3)
    valid_targets.append(valid_targets3)
    invalid_targets.append(invalid_targets3)

    return classes, valid_targets, invalid_targets


if __name__ == "__main__":
    main()
