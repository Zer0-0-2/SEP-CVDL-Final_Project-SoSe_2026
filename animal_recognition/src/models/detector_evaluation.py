## this is a random file to evaluate the detectors
# run it using
# python -m animal_recognition.src.models.detector_evaluation


from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

import animal_recognition.src.models.yoloworld as yoloworld


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ANIMAL_RECOG_DIR = PROJECT_ROOT / "animal_recognition"

DEFAULT_IMAGE_FOLDER = PROJECT_ROOT / "images"
DEFAULT_LABELS_FILE = DEFAULT_IMAGE_FOLDER / "labels.csv"


CONFIDENCE_THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
MODELS = ["yolov8s-worldv2.pt", "yolov8m-worldv2.pt", "yolov8l-worldv2.pt", "yolov8x-worldv2.pt"]

CAT_OR_DOG_LABELS = [i for i in range(0, 20)]
CONFOUNDER_LABEL = -1


def evaluate_image_folder(
    out_image_dir: Path,
    out_csv_path: Path,
    image_folder: Path = DEFAULT_IMAGE_FOLDER,
):

    classes_list, valid_targets_list, make_boxes_but_reject_target, invalid_targets_list = (
        compute_classes()
    )

    labels_path = image_folder / "labels.csv"
    df = pd.read_csv(labels_path)

    results = []
    for model_name in MODELS:
        for single_class, v_targets, i_targets in zip(
            classes_list, valid_targets_list, invalid_targets_list
        ):
            for confidence_threshold in CONFIDENCE_THRESHOLDS:
                logging.info(
                    f"Evaluating detector with confidence threshold {confidence_threshold} and class set {single_class} and model {model_name}"
                )
                image_label_dict = yoloworld.process_dataset(
                    raw_dir=PROJECT_ROOT / "images",
                    processed_dir=ANIMAL_RECOG_DIR / "data" / "processed_yoloworld",
                    rejected_dir=ANIMAL_RECOG_DIR / "data" / "rejected_yoloworld",
                    model_name=model_name,
                    model_confidence_threshold=confidence_threshold,
                    model_classes=single_class,
                    valid_targets=v_targets,
                    invalid_targets=i_targets,
                    make_boxes_but_reject_targets=make_boxes_but_reject_target,
                    test_provided_image_folder=True,
                )

                y_true_list = []
                y_pred_list = []
                for filename, label in df[["filename", "label"]].itertuples(index=False):
                    expected = "catordog" if label != -1 else "notcatordog"
                    predicted = image_label_dict[filename]

                    y_true_list.append(expected)
                    y_pred_list.append(predicted)

                correct = sum(
                    expected == predicted for expected, predicted in zip(y_true_list, y_pred_list)
                )
                total = len(y_true_list)
                accuracy = correct / total if total > 0 else 0

                results.append(
                    {
                        "class_set": str(single_class),
                        "confidence_threshold": confidence_threshold,
                        "accuracy": accuracy,
                    }
                )

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_csv_path, index=False)
    logging.info(f"Evaluation results saved to {out_csv_path}")

    return results_df


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
    make_boxes_but_reject_targets = []
    invalid_targets = []

    # All options
    accept = [
        "cat",
        "dog",
        "domestic cat",
        "domestic dog",
        "sphynx cat",
        "bombay cat",
        "birman cat",
    ]

    make_boxes_but_reject = [
        "tiger",
        "big tiger",
        "predator tiger",
        "fox",
        "wolf",
        "coyote",
    ]

    reject = [
        "drawing",
        "painting",
    ]
    classes.append(accept + make_boxes_but_reject + reject)
    valid_targets.append(accept)
    make_boxes_but_reject_targets.append(make_boxes_but_reject)
    invalid_targets.append(make_boxes_but_reject + reject)

    return classes, valid_targets, make_boxes_but_reject_targets, invalid_targets


if __name__ == "__main__":
    main()
