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
    out_csv_path: Path,
    image_folder: Path = DEFAULT_IMAGE_FOLDER,
):

    classes_list, cutoff_indicies_list = compute_classes()

    labels_path = image_folder / "labels.csv"
    df = pd.read_csv(labels_path)

    results = []
    for model_name in MODELS:
        for single_class, cutoff in zip(classes_list, cutoff_indicies_list):
            model = yoloworld.YoloWorldDetector(
                model_name=model_name,
                model_classes=single_class,
                reject_classes_index=cutoff,
            )
            for confidence_threshold in CONFIDENCE_THRESHOLDS:
                logging.info(
                    f"Evaluating detector with confidence threshold {confidence_threshold} and class set {single_class} and model {model_name}"
                )

                y_true_list = []
                y_pred_list = []
                for filename, label in df[["filename", "label"]].itertuples(index=False):
                    expected = "catordog" if label != -1 else "notcatordog"
                    predicted, confidence, class_id = model.predict(
                        image_folder / filename, confidence_threshold=confidence_threshold
                    )

                    catorodg = "catordog" if predicted is not None else "notcatordog"

                    y_true_list.append(expected)
                    y_pred_list.append(catorodg)

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
        out_csv_path=PROJECT_ROOT / "animal_recognition" / "data" / "evaluation_results.csv",
    )


def compute_classes():
    classes = []
    indexes = []

    # All options
    classes_1 = [
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
    index_1 = classes_1.index("tiger")

    # most barebones
    classes_2 = [
        "cat",
        "dog",
    ]
    index_2 = None

    # only anmals
    classes_3 = [
        "cat",
        "dog",
        "tiger",
    ]
    index_3 = classes_3.index("tiger")

    # no breeds
    classes_4 = [
        "cat",
        "dog",
        "domestic cat",
        "domestic dog",
        "tiger",
        "big tiger",
        "fox",
        "wolf",
        "coyote",
        "drawing",
        "painting",
    ]
    index_4 = classes_4.index("tiger")

    # test only with difficoult breeds
    classes_5 = [
        "cat",
        "dog",
        "domestic cat",
        "domestic dog",
        "sphynx cat",
        "bombay cat",
        "birman cat",  # 6
        "tiger",
    ]
    index_5 = classes_5.index("tiger")
    
    # add the classes and indexes here
    classes.append(classes_1)
    classes.append(classes_2)
    classes.append(classes_3)
    classes.append(classes_4)
    classes.append(classes_5)

    indexes.append(index_1)
    indexes.append(index_2)
    indexes.append(index_3)
    indexes.append(index_4)
    indexes.append(index_5)
    return classes, indexes


if __name__ == "__main__":
    main()
