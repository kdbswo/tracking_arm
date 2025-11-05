from ultralytics import YOLO
import torch
import numpy as np


class Detector:
    def __init__(
        self,
        weights: str = "yolov8n.pt",
        device: str | None = None,
        conf: float = 0.25,
        imgsz: int | None = None,
        use_tracker: bool = True,
        tracker_cfg: str = "botsort.yaml",
        classes: list[int] | None = [0],
    ):
        self.model = YOLO(weights)
        if device is None:
            self.model.to(device)
        elif torch.cuda.is_available():
            self.model.to("cuda")

            try:
                self.model.model.half()
            except Exception:
                pass
        self.conf = conf
        self.imgsz = imgsz
        self.user_tracker = use_tracker
        self.tracker_cfg = tracker_cfg
        self.classes = classes

    def infer(self, frame):
        """
        일반탐지
        frame: BGR np.ndarray (H,W,3) - OpenCV 프레임
        return: results(ultralytics Results)
        """
        res = self.model.predict(
            frame,
            conf=self.conf,
            imgsz=self.imgsz if self.imgsz else None,
            verbose=False,
            classes=self.classes,
        )[0]
        return res

    @staticmethod
    def draw(frame, results, target_id=None):
        import cv2

        boxes = results.boxes
        names = results.names
        if boxes is None or boxes.xyxy is None:
            return frame

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
        clss = boxes.cls.cpu().numpy() if boxes.cls is not None else []
        ids = boxes.id.cpu().numpy() if boxes.id is not None else [None] * len(xyxy)

        for box, c, cls_id, tid in zip(xyxy, confs, clss, ids):
            x1, y1, x2, y2 = map(int, box)
            is_target = (
                target_id is not None and tid is not None and int(tid) == int(target_id)
            )
            color = (0, 128, 255) if is_target else (0, 255, 0)
            label = f"{names[int(cls_id)]} {c:.2f}"
            if tid is not None:
                label += f" id:{int(tid)}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        return frame
