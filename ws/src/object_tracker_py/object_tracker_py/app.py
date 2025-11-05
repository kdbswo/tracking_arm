import time
import cv2
from flask_webcam_client import VideoStreamClient
from detector import Detector


def main():
    server_url = "http://192.168.0.164:5000/stream"

    # 1) 스트림 시작
    client = VideoStreamClient(server_url)
    client.start_stream()

    # 2) 물체 감지기 준비
    detector = Detector(weights="yolov8n.pt", conf=0.4, imgsz=640)

    prev = time.time()
    try:
        # 간단 : 최신 프레임만 뽑아 추론(드롭 허용) - 지연 적음
        while True:
            frame = client.get_latest_frame()
            if frame is None:
                cv2.waitKey(1)
                continue

            results = detector.infer(frame)
            detector.draw(frame, results)

            now = time.time()
            fps = 1.0 / (now - prev) if now > prev else 0.0
            prev = now
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
            cv2.imshow("YOLO Detection (MJPEG Stream)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        client.stop_stream()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
