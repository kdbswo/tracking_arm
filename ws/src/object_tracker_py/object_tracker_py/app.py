import time
import threading
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

    latest_results = None
    result_lock = threading.Lock()
    stop_event = threading.Event()

    def detection_loop():
        nonlocal latest_results
        while not stop_event.is_set():
            frame_for_detection = client.get_latest_frame()
            if frame_for_detection is None:
                time.sleep(0.01)
                continue
            results = detector.infer(frame_for_detection)
            with result_lock:
                latest_results = results

    detection_thread = threading.Thread(target=detection_loop, daemon=True)
    detection_thread.start()

    prev = time.time()
    try:
        # 간단 : 최신 프레임만 뽑아 추론(드롭 허용) - 지연 적음
        while True:
            frame = client.get_latest_frame()
            if frame is None:
                cv2.waitKey(1)
                continue

            with result_lock:
                results = latest_results

            display_frame = frame if results is None else frame.copy()
            if results is not None:
                detector.draw(display_frame, results)

            now = time.time()
            fps = 1.0 / (now - prev) if now > prev else 0.0
            prev = now
            text = f"FPS: {fps:.1f}"
            org = (10, 30)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1.0
            thickness = 2

            # Draw text outline for readability
            cv2.putText(
                display_frame,
                text,
                org,
                font,
                scale,
                (0, 0, 0),
                thickness + 2,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                text,
                org,
                font,
                scale,
                (0, 255, 255),
                thickness,
                lineType=cv2.LINE_AA,
            )
            cv2.imshow("YOLO Detection (MJPEG Stream)", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stop_event.set()
        detection_thread.join(timeout=1.0)
        client.stop_stream()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
