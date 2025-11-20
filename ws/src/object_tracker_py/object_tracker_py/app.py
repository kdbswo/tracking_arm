import math
import threading
import time

import cv2

try:
    from .flask_webcam_client import VideoStreamClient, resolve_stream_url
    from .detector import Detector
    from .arm_cmd_node import init_ros2, send_cmd, send_init_pose, spin_once
except ImportError:  # pragma: no cover - allow running as loose script
    from flask_webcam_client import VideoStreamClient, resolve_stream_url
    from detector import Detector
    from arm_cmd_node import init_ros2, send_cmd, send_init_pose, spin_once

TARGET_ID = None
CLICK_PT = None
_CENTER_STATE = False


def on_mouse(event, x, y, flags, param):
    global CLICK_PT
    if event == cv2.EVENT_LBUTTONDOWN:
        CLICK_PT = (x, y)


def plan_control_command(
    target_center: tuple[float, float], frame_center: tuple[float, float]
) -> tuple[float, float] | None:
    """
    화면 중심 대비 타겟 위치를 -1.0~1.0 범위로 정규화해 좌우 명령을 만든다.
    중앙 구간에서는 히스테리시스(안쪽 0.2, 바깥 0.25)를 적용해 흔들림을 줄인다.

    Returns: (ex, 0) where ex<0=왼쪽, ex>0=오른쪽. None if inputs invalid.
    """
    if not target_center or not frame_center:
        return None

    tx, _ty = target_center
    cx, _cy = frame_center
    if cx == 0:  # 안전 가드
        return None

    # 화면 가로 중심을 0으로, 좌 -1.0, 우 +1.0으로 정규화
    ex = (tx - cx) / cx
    ex = max(-1.0, min(1.0, ex))

    # 중앙 구간 히스테리시스: 안쪽 0.1로 진입, 바깥 0.15로 해제
    global _CENTER_STATE
    enter_th = 0.1
    exit_th = 0.15
    if _CENTER_STATE:
        if abs(ex) <= exit_th:
            ex = 0.0
        else:
            _CENTER_STATE = False
    else:
        if abs(ex) <= enter_th:
            _CENTER_STATE = True
            ex = 0.0

    return (ex, 0.0)


def dispatch_control_command(command: tuple[float, float] | None) -> None:
    """Placeholder hook that will eventually deliver commands to the arm."""
    if not command:
        return
    pan_cmd, tilt_cmd = command
    send_cmd(pan_cmd, tilt_cmd)


def main():
    global TARGET_ID, CLICK_PT

    init_ros2()
    server_url = resolve_stream_url()
    client = VideoStreamClient(server_url)
    client.start_stream()

    pose_cmd = [0.0, 0.0, 0.0, 0.0, 90.0, 0.0]

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
                command = plan_control_command(target_center, (cx_ref, cy_ref))
                if command is not None:
                    dispatch_control_command(command)
            else:
                # 타깃이 사라졌으면 즉시 정지 명령 전송
                dispatch_control_command((0.0, 0.0))

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
