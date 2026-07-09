from ultralytics import YOLO

'''
To test this file:
bash
python3 -m animal-recognition.src.models.detector
'''


#Target classes
CAT_CLASS = 15
DOG_CLASS = 16


class AnimalDetector:
    def __init__(self, weights: str = "yolov8m.pt"):
        self.model = YOLO(weights) #auto-downloads for the first run

    def _parse_results(self, results) -> list:
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                detections.append({"class_id": cls_id, "confidence": conf, "box": xyxy})
        return detections

    def detect(self, image_path: str) -> list:
        """Run detection on a file path, return list of {class_id, confidence, box}."""
        results = self.model(image_path, classes=[CAT_CLASS, DOG_CLASS], verbose=False)
        return self._parse_results(results)

    def detect_pil(self, image) -> list:
        """Run detection on a PIL image, return list of {class_id, confidence, box}."""
        results = self.model(image, classes=[CAT_CLASS, DOG_CLASS], verbose=False)
        return self._parse_results(results)
    
def draw_detections(image_path: str, detections: list, out_path: str = "test_images/annotated.jpg"):
    """Draw bounding boxes + labels on the image and save it for visual inspection."""
    import cv2

    img = cv2.imread(image_path)
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        label = "cat" if d["class_id"] == CAT_CLASS else "dog"
        color = (0, 255, 0) #green, BGR
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {d['confidence']:.2f}"
        cv2.putText(img, text, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imwrite(out_path, img)
    print(f"Saved annotated image to {out_path}")


if __name__ == "__main__":
    detector = AnimalDetector()
    image_path = "test_images/test1.jpg"
    dets = detector.detect(image_path)

    detector = AnimalDetector()
    dets = detector.detect("test_images/test1.jpg")
    print(f"Found {len(dets)} detection(s):")
    for d in dets:
        label = "cat" if d["class_id"] == CAT_CLASS else "dog"
        print(f" {label}: confidence = {d['confidence']:.3f}, box={d['box']}")
    
    if dets: 
        draw_detections(image_path, dets)