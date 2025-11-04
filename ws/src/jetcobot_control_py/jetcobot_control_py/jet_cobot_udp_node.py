import socket  # UDP 전송을 위해 사용
import threading  # 영상 전송을 별도 스레드로 처리하기 위해 사용
import time  # 전송 주기 제어를 위한 sleep 호출에 사용
from typing import Optional  # main 함수 인자를 명시하기 위한 타입힌트
from pymycobot.mycobot280 import MyCobot280  # MyCobot 280 본체 접속을 위해 사용

import cv2  # OpenCV: 카메라 프레임 캡처 및 JPEG 인코딩 담당
import rclpy  # ROS 2 파이썬 클라이언트 라이브러리
from rclpy.node import Node  # ROS 2 노드 기반 클래스


class JetcobotUdpNode(Node):
    """카메라 프레임을 캡처해 JPEG으로 압축하고 UDP로 송출하는 ROS 2 노드."""

    def __init__(self) -> None:
        super().__init__("jetcobot_udp_node")  # 노드 이름을 지정하며 부모 초기화

        self.mc = MyCobot280("/dev/ttyJETCOBOT", 1_000_000)  # MyCobot과 시리얼 연결 시도
        self.mc.thread_lock = True  # pymycobot 내부에서 스레드 충돌을 막도록 설정
        self.get_logger().info("로봇이 연결되었습니다.")  # 연결 성공 로그 출력

        # UDP 송출에 사용할 파라미터 등록
        self.declare_parameter("udp_host", "192.168.0.17")  # 영상 수신 PC IP
        self.declare_parameter("udp_port", 6240)  # 수신 PC UDP 포트
        self.declare_parameter("jpeg_quality", 80)  # JPEG 인코딩 품질(기본 80)

        # 영상 캡처에 필요한 카메라 설정 파라미터 등록
        self.declare_parameter("camera_device", "/dev/jetcocam0")  # 라즈베리 카메라 디바이스
        self.declare_parameter("width", 640)  # 캡처 가로 해상도
        self.declare_parameter("height", 480)  # 캡처 세로 해상도
        self.declare_parameter("fps", 30.0)  # 목표 FPS

        # 파라미터 값을 실제 멤버 변수로 읽어온다
        self.udp_host = self.get_parameter("udp_host").get_parameter_value().string_value
        self.udp_port = int(self.get_parameter("udp_port").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)

        # UDP 송출 소켓 생성
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # 카메라 설정값 읽기
        cam_device = self.get_parameter("camera_device").get_parameter_value().string_value
        width = int(self.get_parameter("width").value)
        height = int(self.get_parameter("height").value)
        fps = float(self.get_parameter("fps").value)

        # OpenCV로 카메라 디바이스 초기화
        self.cap = cv2.VideoCapture(cam_device)
        if not self.cap.isOpened():
            self.get_logger().warn(f"카메라를 열 수 없습니다: {cam_device}")
        else:
            if width > 0:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # 가로 해상도 적용
            if height > 0:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # 세로 해상도 적용
            if fps > 0:
                self.cap.set(cv2.CAP_PROP_FPS, fps)  # FPS 적용

        # 스레드 종료를 제어하기 위한 이벤트
        self._stop_event = threading.Event()
        # 영상 송출을 담당할 백그라운드 스레드 시작
        self._video_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._video_thread.start()

    def _stream_loop(self) -> None:
        """카메라 프레임을 읽어 UDP로 끊김 없이 보내는 루프."""
        max_packet = 60_000  # UDP 페이로드 안전 크기(60KB)
        frame_id = 0  # 프레임 순번(재조립용)

        while not self._stop_event.is_set():
            if not self.cap or not self.cap.isOpened():
                time.sleep(0.1)  # 카메라 준비가 안 된 경우 잠시 대기
                continue

            ok, frame = self.cap.read()  # 1 프레임 캡처
            if not ok:
                continue  # 캡처 실패 시 다음 반복

            success, buffer = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],  # JPEG 품질 적용
            )
            if not success:
                self.get_logger().warn("프레임 JPEG 인코딩 실패")
                continue

            data = buffer.tobytes()  # JPEG 결과를 bytes로 변환
            total_packets = max(1, (len(data) + max_packet - 1) // max_packet)  # 필요한 조각 수 계산

            for packet_num in range(total_packets):
                start = packet_num * max_packet  # 조각 시작 위치
                end = min(start + max_packet, len(data))  # 조각 끝 위치
                payload = data[start:end]  # 실제 전송할 조각
                header = f"{frame_id},{packet_num},{total_packets}||".encode("ascii")  # 재조립용 헤더
                packet = header + payload  # 헤더와 데이터를 합쳐 UDP 패킷 구성

                try:
                    self.sock.sendto(packet, (self.udp_host, self.udp_port))  # UDP 송출
                except OSError as exc:
                    self.get_logger().warn(f"UDP 전송 실패: {exc}")
                    break  # 현재 프레임 전송 중단

            frame_id += 1  # 다음 프레임 번호 증가
            time.sleep(0.03)  # 약 30FPS에 해당하는 지연

    def destroy_node(self) -> bool:
        """노드 종료 시 자원 정리."""
        self._stop_event.set()  # 송출 스레드에 종료 신호
        if self._video_thread.is_alive():
            self._video_thread.join(timeout=1.0)  # 스레드 종료 대기

        if self.cap and self.cap.isOpened():
            self.cap.release()  # 카메라 자원 해제
        self.sock.close()  # UDP 소켓 닫기
        return super().destroy_node()


def main(args: Optional[list[str]] = None) -> None:
    """ros2 run으로 호출될 진입점."""
    rclpy.init(args=args)  # ROS 2 초기화
    node = JetcobotUdpNode()  # 노드 인스턴스 생성
    try:
        rclpy.spin(node)  # ROS 이벤트 루프 실행
    except KeyboardInterrupt:
        pass  # Ctrl+C 종료 허용
    finally:
        node.destroy_node()  # 노드 자원 정리
        rclpy.shutdown()  # ROS 2 종료


if __name__ == "__main__":
    main()  # 모듈 직접 실행 시 main 호출
