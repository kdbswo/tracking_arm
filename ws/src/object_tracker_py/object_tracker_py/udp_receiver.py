import socket  # UDP 패킷 수신을 위한 표준 라이브러리
import threading  # 수신을 백그라운드 스레드로 처리하기 위한 모듈
import time  # 버퍼 정리 시 타임아웃 계산에 사용
from typing import Dict, Optional  # 타입 힌트용 Dict/Optional

import cv2  # OpenCV: 수신 프레임 확인 및 디코딩 검증
import numpy as np  # JPEG 바이트를 배열로 변환하기 위해 사용
import rclpy  # ROS 2 파이썬 클라이언트
from rclpy.node import Node  # ROS 2 노드 기본 클래스


class UdpFrameReceiver(Node):
    """UDP로 분할 전송된 JPEG 프레임을 재조립하고 필요 시 화면에 표시하는 노드."""

    def __init__(self) -> None:
        super().__init__("udp_frame_receiver")  # 노드 이름 지정

        # 수신 설정을 파라미터로 선언
        self.declare_parameter("listen_host", "0.0.0.0")  # 바인드할 IP (기본: 모든 인터페이스)
        self.declare_parameter("listen_port", 6240)  # 수신할 UDP 포트
        self.declare_parameter("display", True)  # 수신 프레임을 화면에 띄울지 여부
        self.declare_parameter("max_buffer", 250)  # 보관할 프레임 버퍼 최대 개수

        # 파라미터 값을 실제 변수로 읽어서 저장
        listen_host = self.get_parameter("listen_host").get_parameter_value().string_value
        listen_port = int(self.get_parameter("listen_port").value)
        self._display = bool(self.get_parameter("display").value)
        self._max_buffer = int(self.get_parameter("max_buffer").value)

        # UDP 소켓을 열고 지정한 포트에 바인딩
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((listen_host, listen_port))
        self._sock.settimeout(0.5)  # 수신이 없을 때 주기적으로 깨어나도록 타임아웃 설정

        # 프레임 조각을 임시 저장하는 버퍼 {frame_id: {...}} 구성
        self._buffers: Dict[int, Dict[str, object]] = {}
        self._stop_event = threading.Event()  # 수신 스레드 종료를 위한 이벤트
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)  # 수신 스레드 생성
        self._thread.start()  # 스레드 시작

        self.get_logger().info(f"UDP 수신 대기 중: {listen_host}:{listen_port}")

    def _recv_loop(self) -> None:
        """UDP 패킷을 계속 받아 프레임 조각으로 누적하고 완성되면 처리."""
        while not self._stop_event.is_set():
            try:
                data, _ = self._sock.recvfrom(65_507)  # 한 번에 받을 수 있는 최대 UDP 패킷 크기
            except socket.timeout:
                self._prune_buffers()  # 오래된 버퍼 정리
                continue
            except OSError:
                break  # 소켓이 닫히면 루프 종료

            try:
                header, payload = data.split(b"||", 1)  # 헤더와 JPEG 조각 분리
                frame_id_str, chunk_idx_str, chunk_total_str = header.decode().split(",")
                frame_id = int(frame_id_str)  # 프레임 순번
                chunk_idx = int(chunk_idx_str)  # 현재 조각 번호
                chunk_total = int(chunk_total_str)  # 전체 조각 수
            except ValueError:
                self.get_logger().warn("잘못된 헤더를 수신했습니다. 패킷을 건너뜁니다.")
                continue

            # 프레임 ID 기준으로 버퍼 딕셔너리를 생성하거나 기존 것 사용
            buf = self._buffers.setdefault(
                frame_id,
                {
                    "total": chunk_total,
                    "chunks": {},
                    "last_update": time.monotonic(),
                },
            )
            buf["chunks"][chunk_idx] = payload  # 해당 조각 저장
            buf["last_update"] = time.monotonic()  # 최근 갱신 시각 기록

            if len(buf["chunks"]) == buf["total"]:
                ordered = [buf["chunks"][i] for i in range(buf["total"])]  # 순서대로 조각 정렬
                jpeg_bytes = b"".join(ordered)  # 하나의 JPEG 바이트로 합치기
                self._buffers.pop(frame_id, None)  # 버퍼에서 제거
                self._handle_frame(jpeg_bytes)  # 완성된 프레임 처리

    def _prune_buffers(self) -> None:
        """버퍼가 너무 많아졌거나 오래된 항목을 삭제."""
        if len(self._buffers) <= self._max_buffer:
            return
        now = time.monotonic()
        stale_keys = [
            fid
            for fid, meta in self._buffers.items()
            if now - float(meta.get("last_update", now)) > 1.0  # 1초 이상 업데이트 없으면 삭제
        ]
        for fid in stale_keys:
            self._buffers.pop(fid, None)

    def _handle_frame(self, jpeg_bytes: bytes) -> None:
        """완성된 JPEG 바이트를 OpenCV 프레임으로 디코딩하고 표시."""
        np_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)  # numpy 배열로 변환
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # JPEG → BGR 이미지
        if frame is None:
            self.get_logger().warn("JPEG 디코딩 실패")
            return

        self.get_logger().debug(f"프레임 수신: {frame.shape}")  # 프레임 크기 디버그 출력

        if self._display:
            cv2.imshow("udp_stream", frame)  # 실시간으로 영상 표시
            if cv2.waitKey(1) == 27:  # ESC 입력 시 수신 종료
                self.get_logger().info("ESC 입력 감지, 수신 종료")
                self._stop_event.set()

        # 이후 물체 감지/추적 알고리즘을 적용하려면 이곳에서 frame을 활용하면 됨

    def destroy_node(self) -> bool:
        """노드 종료 시 백그라운드 스레드와 리소스를 정리."""
        self._stop_event.set()  # 수신 루프 종료 요청
        if hasattr(self, "_sock"):
            self._sock.close()  # UDP 소켓 닫기
        if hasattr(self, "_thread") and self._thread.is_alive():
            self._thread.join(timeout=1.0)  # 스레드 종료 대기
        cv2.destroyAllWindows()  # OpenCV 창 정리
        return super().destroy_node()


def main(args: Optional[list[str]] = None) -> None:
    """ros2 run object_tracker_py udp_receiver로 실행되는 진입점."""
    rclpy.init(args=args)
    node = UdpFrameReceiver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
