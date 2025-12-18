import numpy as np
from typing import List, Tuple

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class YOLODetector:
    """
    YOLOv8 detector wrapper.

    Detects poultry-like objects using multiple COCO classes
    because broiler chickens are not reliably labeled as 'bird' only.

    Output format:
    (x1, y1, x2, y2, confidence)
    """

    # COCO classes that often match poultry
    POULTRY_CLASSES = {14, 15, 16}  # bird, cat, dog (proxy workaround)

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_thresh: float = 0.10
    ):
        if YOLO is None:
            raise ImportError(
                "ultralytics not installed. Run: pip install ultralytics"
            )

        self.conf_thresh = conf_thresh
        self.model = YOLO(model_name)

    def detect(
        self, frame: np.ndarray
    ) -> List[Tuple[float, float, float, float, float]]:
        """
        Run detection and return bounding boxes.

        Returns:
        [
            (x1, y1, x2, y2, confidence),
            ...
        ]
        """

        results = self.model(frame, verbose=False)
        if not results:
            return []

        res = results[0]
        boxes = res.boxes
        if boxes is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy()

        detections = []

        for box, conf, cls_id in zip(xyxy, confs, clss):
            if conf < self.conf_thresh:
                continue

            # Accept multiple animal-like classes
            if int(cls_id) in self.POULTRY_CLASSES:
                x1, y1, x2, y2 = box.tolist()
                detections.append(
                    (x1, y1, x2, y2, float(conf))
                )

        return detections
