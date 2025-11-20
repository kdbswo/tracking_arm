import math
import threading
import time

import cv2

try:
    from .flask_webcam_client import VideoStreamClient, resolve_stream_url
    from .detector import Detector
    from .arm_cmd_node import (
        init_ros2,
        send_init_pose,
        send_target_px,
        spin_once,
    )
except ImportError:  # pragma: no cover - allow running as loose script
    from flask_webcam_client import VideoStreamClient, resolve_stream_url
    from detector import Detector
    from arm_cmd_node import init_ros2, send_init_pose, send_target_px, spin_once

TARGET_ID = None
CLICK_PT = None


def on_mouse(event, x, y, flags, param):
    global CLICK_PT
    if event == cv2.EVENT_LBUTTONDOWN:
        CLICK_PT = (x, y)


def main():
    global TARGET_ID, CLICK_PT

    init_ros2()
    server_url = resolve_stream_url()
    client = VideoStreamClient(server_url)
    client.start_stream()

    pose_cmd = [0.0, 0.0, 0.0, 0.0, 45.0, 0.0]

    detector = Detector(
        weights="yolov8n.pt",
        conf=0.4,
        imgsz=640,
        use_tracker=True,
        tracker_cfg="botsort.yaml",
        classes=[0],
    )

    latest_results = None
    result_lock = threading.Lock()
    stop_event = threading.Event()

    def detection_loop():
        nonlocal latest_results
        while not stop_event.is_set():
            frame = client.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            res = detector.track(frame)
            with result_lock:
                latest_results = res

    worker = threading.Thread(target=detection_loop, daemon=True)
    worker.start()

    cv2.namedWindow("FOLLOW PERSON")
    cv2.setMouseCallback("FOLLOW PERSON", on_mouse)

    last_sent_target = False

    try:
        while True:
            frame = client.get_latest_frame()
            if frame is None:
                cv2.waitKey(1)
                spin_once()
                continue

            H, W = frame.shape[:2]
            cx_ref, cy_ref = W / 2.0, H / 2.0

            with result_lock:
                results = latest_results

            target_center = None
            boxes = results.boxes if results is not None else None

            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.int().cpu().numpy()
                clss = boxes.cls.int().cpu().numpy()
                confs = (
                    boxes.conf.cpu().numpy()
                    if boxes.conf is not None
                    else [0.0] * len(xyxy)
                )

                if CLICK_PT is not None:
                    x_click, y_click = CLICK_PT
                    CLICK_PT = None
                    best_id, best_d = None, 1e9
                    for bb, tid, cls in zip(xyxy, ids, clss):
                        if results.names[int(cls)] != "person":
                            continue
                        x1, y1, x2, y2 = map(int, bb)
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        d = math.hypot(cx - x_click, cy - y_click)
                        if d < best_d:
                            best_d, best_id = d, int(tid)
                    if best_id is not None:
                        TARGET_ID = best_id
                        print(f"[LOCK] TARGET_ID={TARGET_ID}")

                if TARGET_ID is not None:
                    for bb, tid in zip(xyxy, ids):
                        if int(tid) == int(TARGET_ID):
                            x1, y1, x2, y2 = map(int, bb)
                            target_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                            break

            display = frame.copy()
            if results is not None:
                detector.draw(display, results, target_id=TARGET_ID)

            cv2.drawMarker(
                display,
                (int(cx_ref), int(cy_ref)),
                (255, 255, 255),
                markerType=cv2.MARKER_TILTED_CROSS,
                thickness=2,
            )

            if target_center is not None:
                tx, ty = target_center
                cv2.drawMarker(
                    display,
                    (int(tx), int(ty)),
                    (0, 255, 255),
                    markerType=cv2.MARKER_CROSS,
                    thickness=2,
                )
                send_target_px(target_center, frame.shape)
                last_sent_target = True
            else:
                # 타깃이 사라졌으면 한 번만 알림 보내고 로봇 쪽의 타임아웃 로직을 활용
                if last_sent_target:
                    send_target_px(None, None)
                    last_sent_target = False

            cv2.putText(
                display,
                f"TARGET_ID: {TARGET_ID}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                display,
                "Click person to lock. [r]=release [q]=quit [f]=init pose",
                (10, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )

            cv2.imshow("FOLLOW PERSON", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                TARGET_ID = None
                print("[RELEASE] clear target")
            if key == ord("q"):
                break
            if key == ord("f"):
                send_init_pose(pose_cmd)
                print(f"[POSE] init pose {pose_cmd}")

            spin_once()

    finally:
        stop_event.set()
        worker.join(timeout=1.0)
        client.stop_stream()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
