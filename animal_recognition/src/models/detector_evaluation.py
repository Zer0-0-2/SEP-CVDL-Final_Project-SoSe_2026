## this is a random file to evaluate the detectors
# run it using
# python -m animal_recognition.src.models.detector_evaluation


from __future__ import annotations

from pathlib import Path

import logging
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


class DetectorEvaluator:
    def __init__(
        self,
        image_folder: Path = DEFAULT_IMAGE_FOLDER,
        models: list[str] = MODELS,
        confidence_thresholds: list[float] = CONFIDENCE_THRESHOLDS,
        model_classes_indices: list[int] = [
            i for i in range(6)
        ],  # select which classes you want to have from compute_classes
    ):
        self.image_folder = image_folder
        self.models = models
        self.confidence_thresholds = confidence_thresholds
        self.model_classes_indices = model_classes_indices
        self.classes_list, self.cutoff_indices_list = self.compute_classes()

    def compute_classes(self):
        all_classes = []
        all_indexes = []

        # All options
        classes_0 = [
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
        index_0 = classes_0.index("tiger")
        all_classes.append(classes_0)
        all_indexes.append(index_0)

        # most barebones
        classes_1 = [
            "cat",
            "dog",
        ]
        index_1 = None
        all_classes.append(classes_1)
        all_indexes.append(index_1)

        # only anmals
        classes_2 = [
            "cat",
            "dog",
            "tiger",
        ]
        index_2 = classes_2.index("tiger")
        all_classes.append(classes_2)
        all_indexes.append(index_2)

        # no breeds
        classes_3 = [
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
        index_3 = classes_3.index("tiger")
        all_classes.append(classes_3)
        all_indexes.append(index_3)

        # test only with difficoult breeds
        classes_4 = [
            "cat",
            "dog",
            "domestic cat",
            "domestic dog",
            "sphynx cat",
            "bombay cat",
            "birman cat",  # 6
            "tiger",
        ]
        index_4 = classes_4.index("tiger")

        all_classes.append(classes_4)
        all_indexes.append(index_4)

        # all options without painting
        classes_5 = [
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
        ]
        index_5 = classes_5.index("tiger")
        all_classes.append(classes_5)
        all_indexes.append(index_5)

        classes = [all_classes[i] for i in self.model_classes_indices]
        indices = [all_indexes[i] for i in self.model_classes_indices]

        return classes, indices

    def evaluate(
        self,
        out_csv_path: Path,
    ):

        labels_path = self.image_folder / "labels.csv"
        df = pd.read_csv(labels_path)

        results = []
        for model_name in self.models:
            for single_class, cutoff in zip(self.classes_list, self.cutoff_indices_list):
                model = yoloworld.YoloWorldDetector(
                    model_name=model_name,
                    model_classes=single_class,
                    reject_classes_index=cutoff,
                )
                cutoff_class = single_class[cutoff] if cutoff is not None else None
                logging.info(
                    f"Evaluating model {model_name} with class set {single_class}, cutoff {cutoff_class} for multiple thresholds: {self.confidence_thresholds}"
                )

                y_true_list = []
                y_pred_lists = {t: [] for t in self.confidence_thresholds}

                for filename, label in df[["filename", "label"]].itertuples(index=False):
                    expected = "catordog" if label != -1 else "notcatordog"
                    y_true_list.append(expected)

                    img_path = self.image_folder / filename
                    preds_dict = model.predict_multiple_thresholds(
                        img_path, confidence_thresholds=self.confidence_thresholds
                    )

                    for t in self.confidence_thresholds:
                        predicted, _, _ = preds_dict[t]
                        catorodg = "catordog" if predicted is not None else "notcatordog"
                        y_pred_lists[t].append(catorodg)

                for confidence_threshold in self.confidence_thresholds:
                    y_pred_list = y_pred_lists[confidence_threshold]

                    tp = sum(
                        1
                        for e, p in zip(y_true_list, y_pred_list)
                        if e == "catordog" and p == "catordog"
                    )
                    tn = sum(
                        1
                        for e, p in zip(y_true_list, y_pred_list)
                        if e == "notcatordog" and p == "notcatordog"
                    )
                    fp = sum(
                        1
                        for e, p in zip(y_true_list, y_pred_list)
                        if e == "notcatordog" and p == "catordog"
                    )
                    fn = sum(
                        1
                        for e, p in zip(y_true_list, y_pred_list)
                        if e == "catordog" and p == "notcatordog"
                    )
                    total = len(y_true_list)
                    accuracy = (tp + tn) / total if total > 0 else 0

                    # technically we want to maximize the recall, since our second model in the pipeline wil handle the actual classification.
                    # the other metrics are just for information to make sure the given configuration is not too bad in tother regards.
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    f1_score = (
                        2 * (precision * recall) / (precision + recall)
                        if (precision + recall) > 0
                        else 0
                    )

                    results.append(
                        {
                            "model_name": model_name,
                            "class_set": str(single_class),
                            "cutoff": cutoff_class,
                            "confidence_threshold": confidence_threshold,
                            "accuracy": accuracy,
                            "true positive": tp,
                            "true negative": tn,
                            "false postivie": fp,
                            "false negative": fn,
                            "precision": precision,
                            "recall": recall,
                            "f1_score": f1_score,
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
    """
    evaluator1 = DetectorEvaluator()
    evaluator1.evaluate(
        out_csv_path=PROJECT_ROOT / "evaluation_folder" / "detector" / "yoloworld_results_full.csv"
    )
    """
    evaluator2 = DetectorEvaluator(
        models=["yolov8l-worldv2.pt", "yolov8x-worldv2.pt"],
        confidence_thresholds=[
            0.001,
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            0.06,
            0.07,
            0.08,
            0.09,
            0.1,
            0.125,
            0.15,
            0.175,
            0.2,
            0.25,
        ],
    )

    evaluator2.evaluate(
        out_csv_path=PROJECT_ROOT
        / "evaluation_folder"
        / "detector"
        / "yoloworld_results_full_more_thresholds.csv"
    )


if __name__ == "__main__":
    main()
