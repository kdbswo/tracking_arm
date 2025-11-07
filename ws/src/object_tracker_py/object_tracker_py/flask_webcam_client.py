import cv2
import numpy as np
import requests
import threading
import time


class VideoStreamClient:
    def __init__(self, server_url):
        """
        비디오 스트림 클라이언트 초기화

        Args:
            server_url (str): 서버스트림 URL (예: 'http://192.168.5.1:5000/stream')
        """
        self.server_url = server_url
        self.stream_active = False
        self.stream_thread = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()

    def start_stream(self):
        """스트림 수신 시작"""
        if self.stream_active:
            print("스트림이 이미 활성화되어 있습니다.")
            return

        self.stream_active = True
        self.stream_thread = threading.Thread(target=self._receive_stream)
        self.stream_thread.daemon = True
        self.stream_thread.start()

    def stop_stream(self):
        """스트림 수신 중지"""
        self.stream_active = False
        if self.stream_thread:
            self.stream_thread.join()
            self.stream_thread = None

    def get_latest_frame(self):
        """최신 프레임 반환"""

        with self.frame_lock:
            return self.latest_frame

    def _receive_stream(self):
        """비디오 스트림 수신 및 처리하는 내부 메서드"""
        try:
            # 스트림에 연결
            response = requests.get(self.server_url, stream=True, timeout=(3.0, 10.0))

            if response.status_code != 200:
                print(f"스트림 연결 실패: 상태 코드 {response.status_code}")
                self.stream_active = False
                return
            # MJPEG 스트림의 바이트 데이터
            bytes_buffer = bytearray()
            soi = b"\xff\xd8"
            eoi = b"\xff\xd9"

            # 스트림 처리 루프
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue

                bytes_buffer.extend(chunk)

                start = bytes_buffer.find(soi)
                end = bytes_buffer.find(eoi, start + 2 if start != -1 else 0)

                while start != -1 and end != -1 and end > start:
                    frame_bytes = bytes(bytes_buffer[start : end + 2])
                    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

                    if frame is not None:
                        with self.frame_lock:
                            self.latest_frame = frame

                    del bytes_buffer[: end + 2]
                    start = bytes_buffer.find(soi)
                    end = bytes_buffer.find(eoi, start + 2 if start != -1 else 0)

                # 스트림이 비활성화되면 종료
                if not self.stream_active:
                    break

        except Exception as e:
            print(f"스트림 수신 중 오류 발생: {e}")
        finally:
            self.stream_active = False


def main():
    server_url = "http://192.168.0.165:5000/stream"

    client = VideoStreamClient(server_url)
    client.start_stream()

    try:
        while True:
            frame = client.get_latest_frame()
            if frame is not None:
                cv2.imshow("Jetcobot Camera Stream", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("종료 중...")
    finally:
        client.stop_stream()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
