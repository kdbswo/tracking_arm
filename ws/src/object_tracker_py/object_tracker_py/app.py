import time, threading, math
import cv2

try:
    from .flask_webcam_client import VideoStreamClient, resolve_stream_url
    from .detector import Detector
    from .arm_cmd_node import init_ros2, spin_once, send_cmd, send_init_pose
except ImportError:  # pragma: no cover - allow running as a loose script
    from flask_webcam_client import VideoStreamClient, resolve_stream_url
    from detector import Detector
    from arm_cmd_node import init_ros2, spin_once, send_cmd, send_init_pose

TARGET_ID = None
CLICK_PT = None


def on_mouse(event, x, y, flags, param):
    global CLICK_PT
    if event == cv2.EVENT_LBUTTONDOWN:
        CLICK_PT = (x, y)


class PID:
    def __init__(self, kp=0.01, ki=0.0, kd=0.0015):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.ie = 0.0
        self.pe = 0.0

    def step(self, e, dt):
        self.ie += e * dt
        de = (e - self.pe) / dt if dt > 0 else 0.0
        self.pe = e
        return self.kp * e + self.ki * self.ie + self.kd * de


def control_robot_arm(pan_cmd, tilt_cmd):
    send_cmd(pan_cmd, tilt_cmd)
    spin_once()


def main():
    global TARGET_ID, CLICK_PT

    init_ros2()
    server_url = resolve_stream_url()
    client = VideoStreamClient(server_url)
    client.start_stream()

    pose_cmd = [0.0, 0.0, 0.0, 0.0, 90.0, -45.0]

    detector = Detector(
        weights="yolov8n.pt",
        conf=0.4,
        imgsz=640,
        use_tracker=True,
        tracker_cfg="botsort.yaml",
        classes=[0],
    )  # 사람만

    latest_results = None
    result_lock = threading.Lock()
    stop_event = threading.Event()

    # 추론(추적) 스레드
    def detection_loop():
        nonlocal latest_results
        while not stop_event.is_set():
            f = client.get_latest_frame()
            if f is None:
                time.sleep(0.01)
                continue
            res = detector.track(f)  # ← predict 대신 track
            with result_lock:
                latest_results = res

    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()

    pid_x = PID(kp=0.003, kd=0.0015)
    pid_y = PID(kp=0.003, kd=0.0015)

    prev = time.time()
    cv2.namedWindow("FOLLOW PERSON")
    cv2.setMouseCallback("FOLLOW PERSON", on_mouse)

    try:
        while True:
            frame = client.get_latest_frame()
            if frame is None:
                cv2.waitKey(1)
                continue

            H, W = frame.shape[:2]
            cx_ref, cy_ref = W // 2, H // 2

            with result_lock:
                results = latest_results

            # 타깃 선택(클릭 시 가장 가까운 사람 ID 할당)
            if (
                results is not None
                and CLICK_PT is not None
                and results.boxes is not None
            ):
                x_click, y_click = CLICK_PT  # 최근 마우스 클릭 좌표 복사
                CLICK_PT = None  # 한 번 처리한 클릭 좌표는 초기화
                if results.boxes.id is not None:
                    xyxy = (
                        results.boxes.xyxy.cpu().numpy()
                    )  # 각 감지 박스의 [x1,y1,x2,y2]
                    ids = results.boxes.id.int().cpu().numpy()  # 추적 ID 배열
                    clss = (
                        results.boxes.cls.int().cpu().numpy()
                    )  # 클래스 인덱스 배열 (예: person=0)
                    best_id, best_d = None, 1e9  # 가장 가까운 박스 정보 초기값
                    for bb, tid, cls in zip(xyxy, ids, clss):
                        if (
                            results.names[int(cls)] != "person"
                        ):  # 사람 클래스만 대상으로 삼음
                            continue
                        x1, y1, x2, y2 = map(int, bb)  # 박스 좌표 정수화
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # 박스 중심 계산
                        d = math.hypot(cx - x_click, cy - y_click)  # 클릭 위치와의 거리
                        if d < best_d:
                            best_d, best_id = d, int(tid)  # 더 가까우면 후보 갱신
                    if best_id is not None:
                        TARGET_ID = best_id  # 타깃 ID 잠금
                        print(f"[LOCK] TARGET_ID={TARGET_ID}")

            # 제어: 타깃이 있으면 중심 오차 → PID → 로봇팔 명령
            target_center = None
            if (
                results is not None
                and results.boxes is not None
                and results.boxes.id is not None
                and TARGET_ID is not None
            ):
                xyxy = results.boxes.xyxy.cpu().numpy()
                ids = results.boxes.id.int().cpu().numpy()
                for bb, tid in zip(xyxy, ids):
                    if int(tid) == int(TARGET_ID):
                        x1, y1, x2, y2 = map(int, bb)
                        target_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        break

            # 시각화
            display = frame.copy()
            if results is not None:
                detector.draw(display, results, target_id=TARGET_ID)

            # 중심/타깃 마커
            cv2.drawMarker(
                display,
                (cx_ref, cy_ref),
                (255, 255, 255),
                markerType=cv2.MARKER_TILTED_CROSS,
                thickness=2,
            )
            if target_center is not None:
                tx, ty = target_center
                cv2.drawMarker(
                    display,
                    (tx, ty),
                    (0, 255, 255),
                    markerType=cv2.MARKER_CROSS,
                    thickness=2,
                )
                # PID
                now = time.time()  # 현재 시각
                dt = max(1e-3, now - prev)  # 지난 루프 대비 경과시간 (최소값 보장)
                prev = now  # 다음 루프를 위해 현재 시각 저장
                ex = cx_ref - tx  # +면 오른쪽으로 팬해야 함
                ey = cy_ref - ty  # +면 위로 틸트해야 함
                pan_cmd = pid_x.step(ex, dt)  # 팬축 PID 제어 출력
                tilt_cmd = pid_y.step(ey, dt)  # 틸트축 PID 제어 출력
                control_robot_arm(pan_cmd, tilt_cmd)  # 계산된 명령을 로봇팔에 전달

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
                print(f"[RELEASE] init pose {pose_cmd}")

            # Keep ROS callbacks (e.g., /arm/state) flowing even when no cmd publish
            spin_once()

    finally:
        stop_event.set()
        t.join(timeout=1.0)
        client.stop_stream()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
