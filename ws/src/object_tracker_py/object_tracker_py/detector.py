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

    def infer(self, frame):
        """
        frame: BGR np.ndarray (H,W,3) - OpenCV 프레임
        return: results(ultralytics Results)
        """
        res = self.model.predict(
            frame,
            conf=self.conf,
            imgsz=self.imgsz if self.imgsz else None,
            verbose=False,
        )[0]
        return res

    @staticmethod
    def draw(frame, results):
        """
        결과를 frame 위에 그려서 반환(BGR in-place)
        """
        boxes = results.boxes
        names = results.names
        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else np.empty((0, 4))
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
        clss = boxes.cls.cpu().numpy() if boxes.cls is not None else []

        for box, c, cls_id in zip(xyxy, confs, clss):
            x1, y1, x2, y2 = map(int, box)
            label = f"{names[int(cls_id)]} {c: .2f}"
            # 사람 클래스만 다른 색으로
            color = (0, 255, 0) if names[int(cls_id)] != "person" else (0, 128, 255)
            thickness = 2
            import cv2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
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
