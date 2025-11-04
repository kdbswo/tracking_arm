import os
import time

import rclpy
from rclpy.node import Node
from pymycobot import MyCobot


class MotionTestNode(Node):
    def __init__(self):
        super().__init__("motion_test_node")
        self.declare_parameter("port", "/dev/ttyJETCOBOT")
        self.declare_parameter("baud", 115200)
        self.declare_parameter("speed", 30)

        self.port = self.get_parameter("port").value
        self.baud = int(self.get_parameter("baud").value)
        self.speed = int(self.get_parameter("speed").value)

        # 간단한 각도 시퀀스로 왕복 동작 확인
        self.sequence = [
            [0, 0, 0, 0, 0, 0],
            [10, -15, 20, 0, 5, 0],
            [-10, 15, -20, 0, -5, 0],
        ]
        self.seq_idx = 0
        self.mc = None
        self._connect()

        self.timer = self.create_timer(3.0, self._tick)
        self.warn_timer = self.create_timer(2.0, self._warn_and_retry)

    def _connect(self):
        if self.mc is not None:
            return

        try:
            self.mc = MyCobot(self.port, self.baud)
            self.get_logger().info(f"MyCobot 연결 완료: {self.port} @ {self.baud}bps")
        except Exception as exc:
            self.mc = None
            self.get_logger().warn(f"MyCobot 연결 실패 ({self.port}): {exc}")

    def _tick(self):
        if self.mc is None:
            return

        target = self.sequence[self.seq_idx]
        try:
            self.mc.send_angles(target, self.speed)
            self.get_logger().info(f"동작 확인용 각도 전송: {target}")
            self.seq_idx = (self.seq_idx + 1) % len(self.sequence)
        except Exception as exc:
            self.get_logger().warn(f"각도 전송 실패: {exc}")
            self._reset_connection()

    def _reset_connection(self):
        if self.mc:
            try:
                self.mc.release_all_servos()
            except Exception:
                pass
        self.mc = None
        time.sleep(0.5)

    def _warn_and_retry(self):
        if self.mc is not None:
            return

        self.get_logger().warn(
            f"MyCobot가 {self.port}에 연결되지 않았습니다. 케이블/전원을 확인해 주세요."
        )

        if os.path.exists(self.port):
            self.get_logger().info("포트를 감지했습니다. 재연결을 시도합니다.")
            self._connect()


def main():
    rclpy.init()
    node = MotionTestNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
