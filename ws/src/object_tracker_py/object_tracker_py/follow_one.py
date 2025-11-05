import cv2, time, math
import torch
from ultralytics import YOLO

TARGET_ID = None
CLICK_PT = None
SELECT_ON_CLICK = True

def on_mouse(event, x, y, flags, param):
    global CLICK_PT
    if event == cv2.EVENT_LBUTTONDOWN:
        CLICK_PT = (x, y)

class PID:
    def __init__(self, kp=0.01, ki=0.0, kd=0.003):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.ie = 0.0
        self.pe = 0.0
    
    def step(self, e, dt):
        self.ie += e * dt
        de = (e - self.pe) / dt if dt > 0 else 0.0
        self.pe = e
        return self.kp*e + self.ki*self.ie + self.kd*de

def control_camera(pan_cmd, tilt_cmd):
    """실제 PTZ/짐벌 제어 자리
    예: 시리얼/PWM/ROS2 토픽 등을 명령 전송.
    여기선 데모로 프린트만
    """
    print(f"[CTRL] pan:{pan_cmd:+.3f}, tilt:{tilt_cmd:+.3f}")

def main():
    global TARGET_ID, CLICK_PT

    # 1) 모델 & 추적기 준비
    model = YOLO("yolov8n.pt")
    device = 0 if torch.cuda.is_available() else "cpu"
    tracker_cfg = 'botsort.yaml'
    CONF, IMG = 0.4, 640

    # 2) 비디오 소스: MJPEG tntls vmfpdladmfh rycprksmd
    # 데모로 RTSP/웹캠/파일 중 하나 선택
    source = 0 # 웹캠.  네트워크 프레임이면 loop에서 frame=client.get_latest_frame()로 교체
    cap = cv2.VideoCapture(source)

    # PID 두 축(가로=pan, 세로=tilt)
    pid_x = PID(kp=0.003, ki=0.0, kd=0.0015)
    pid_y = PID(kp=0.003, ki=0.0, kd=0.0015)

    prev = time.time()
    cv2.namedWindow("FOLLOW-ONE")
    cv2.setMouseCallback("FOLLOW-ONE", on_mouse)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        H, W = frame.shape[:2]
        cx_ref, cy_ref = W // 2, H // 2

        now = time.time()
        dt = now - prev if now > prev else 0.033
        prev = now

        # 3) 추적 호출 (persist=True로 이전 상태 유지)
        results = model.track(
            source=frame,
            conf=CONF,
            imgsz=IMG,
            device=device,
            tracker=tracker_cfg,
            persist=True,
            classes=[0],  # 사람 클래스만
            verbose=False,
        )[0]

        # 모든 트랙 박스와 ID
        boxes = results.boxes
        names = results.names
        target_bbox = None
        target_center = None

        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.int().cpu().numpy()
            clss = boxes.cls.int().cpu().numpy()

            # 4) 클릭으로 타겟 ID 선택
            if SELECT_ON_CLICK and CLICK_PT is not None:
                x_click, y_click = CLICK_PT
                CLICK_PT = None
                # 클릭점에 가장 가까운 박스를 타겟으로
                best_id, best_d = None, 1e9
                for bb, tid, cls in zip(xyxy, ids, clss):
                    if names[int(cls)] != "person":
                        continue
                    x1, y1, x2, y2 = map(int, bb)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    d = math.hypot(cx - x_click, cy - y_click)
                    if d < best_d:
                        best_d = d
                        best_id = int(tid)
                if best_id is not None:
                    TARGET_ID = best_id
                    print(f"[LOCK] TARGET_ID = {TARGET_ID}")

                for bb, tid, cls in zip(xyxy, ids, clss):
                    x1, y1, x2, y2 = map(int, bb)
                    label_id = int(tid)
                    color = (100, 255, 100) if label_id == TARGET_ID else (128, 128, 128)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"id:{label_id}",
                        (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

                    if label_id == TARGET_ID:
                        target_bbox = (x1, y1, x2, y2)
                        target_center = ((x1 + x2) // 2, (y1 + y2) // 2)

            # 5) 타겟 박스가 정해졌다면 그 박스만 추출
            for bb, tid, cls in zip(xyxy, ids, clss):
                x1, y1, x2, y2 = map(int, bb)
                label_id = int(tid)
                color = (100, 255, 100) if label_id == TARGET_ID else (128, 128, 128)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"id:{label_id}",
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

                if label_id == TARGET_ID:
                    target_bbox = (x1, y1, x2, y2)
                    target_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        # 6) 타겟이 정해졌으면 중심 정렬(PID)로 카메라 제아
        if target_center is not None:
            tx, ty = target_center
            cv2.drawMarker(frame, (tx,ty), (0,255,255), markerType=cv2.MARKER_CROSS, thickness=2)
            # 에러(목표-현재)
            ex = cx_ref - tx
            ey = cy_ref - ty
            pan_cmd = pid_x.step(ex, dt)
            tilt_cmd = pid_y.step(ey, dt)
            control_camera(pan_cmd, tilt_cmd)

        # 중심 가이드
        cv2.drawMarker(frame, (cx_ref,cy_ref), (255,255,255), markerType=cv2.MARKER_TILTED_CROSS, thickness=2)
        cv2.putText(frame, f"TARGET_ID: {TARGET_ID}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, "Click person to lock target. [r]=release [q]=quit", (10, H-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        cv2.imshow("FOLLOW-ONE", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'): #타겟 해제
            TARGET_ID = None
            print("[RELEASE] TARGET cleared")
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
