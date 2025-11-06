import cv2
import threading
import time
from flask import Flask, Response

app = Flask(__name__)

# 글로벌 변수로 최신 프레임 저장
frame = None
frame_lock = threading.Lock()


def capture_frames():
    """카메라에서 프레임을 지속적으로 캡처하는 함수"""
    global frame

    # /dev/jetcocam0 카메라 디바이스 열기
    camera = cv2.VideoCapture("/dev/jetcocam0")

    if not camera.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    frame_count = 0
    last_log = time.time()

    while True:
        success, img = camera.read()
        if not success:
            print("프레임을 읽을 수 없습니다.")
            time.sleep(0.1)
            continue

        # 프레임 인코딩 및 저장
        _, buffer = cv2.imencode(".jpg", img)
        with frame_lock:
            frame = buffer.tobytes()

        frame_count += 1
        now = time.time()
        if now - last_log >= 1.0:
            elapsed = now - last_log
            print(f"capture fps: {frame_count / elapsed:.1f}")
            frame_count = 0
            last_log = now

        # 프레임 캡처 간격
        time.sleep(0.02)  # ~30fps


def generate_frames():
    global frame

    while True:
        # 최신 프레임 가져오기
        with frame_lock:
            if frame is None:
                time.sleep(0.1)
                continue
            current_frame = frame

        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + current_frame + b"\r\n"
        )


@app.route("/stream")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def main():
    capture_thread = threading.Thread(target=capture_frames)
    capture_thread.daemon = True
    capture_thread.start()

    app.run(host="0.0.0.0", port=5000, threaded=True)


if __name__ == "__main__":
    main()
