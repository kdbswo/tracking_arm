import os

import rclpy
from geometry_msgs.msg import PointStamped
from pymycobot import MyCobot
from rclpy.node import Node


class FollowNode(Node):
    def __init__(self):
        super().__init__("follow_node")
        self.declare_parameter("port", "/dev/ttyJETCOBOT")
        self.declare_parameter("baud", 115200)
        self.port = self.get_parameter("port").value
        self.baud = int(self.get_parameter("baud").value)

        self.sub = self.create_subscription(PointStamped, "/target_point", self.cb, 10)
        self.mc = None
        self._connect_cobot()

        # 연결이 끊겨 있으면 주기적으로 경고를 띄우고 재시도
        self.warn_timer = self.create_timer(2.0, self._warn_and_retry)

    def cb(self, msg: PointStamped):
        if self.mc is None:
            return

        xyz = [msg.point.x, msg.point.y, msg.point.z]
        # 내장 IK 사용 → 각도 계산 후 전송
        try:
            angles = self.mc.get_angles_from_coords(xyz)
            if angles:
                self.mc.send_angles(angles, 30)
        except Exception as e:
            self.get_logger().warn(f"IK/send fail: {e}")

    def _connect_cobot(self):
        if self.mc is not None:
            return

        try:
            self.mc = MyCobot(self.port, self.baud)
            self.get_logger().info(f"MyCobot 연결 완료: {self.port} @ {self.baud}bps")
        except Exception as e:
            self.mc = None
            self.get_logger().warn(f"MyCobot 연결 실패 ({self.port}): {e}")

    def _warn_and_retry(self):
        if self.mc is not None:
            return

        self.get_logger().warn(
            f"MyCobot가 {self.port}에 연결되지 않았습니다. 케이블/전원을 확인해 주세요."
        )

        if os.path.exists(self.port):
            self.get_logger().info("포트를 감지했습니다. MyCobot 재연결 시도 중...")
            self._connect_cobot()


def main():
    rclpy.init()
    rclpy.spin(FollowNode())
    rclpy.shutdown()
