from __future__ import annotations

from typing import List, Optional

import rclpy
from pymycobot import MyCobot
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class ArmFollowerNode(Node):
    """Minimal bridge: subscribe to /arm/cmd (angles) and forward to MyCobot."""

    def __init__(self) -> None:
        super().__init__("arm_follower_node")

        # 하드웨어 접속 / 업데이트 주기를 파라미터로 노출
        self.declare_parameter("port", "/dev/ttyJETCOBOT")
        self.declare_parameter("baud", 1_000_000)
        self.declare_parameter("speed", 40)
        self.declare_parameter("angle_count", 6)
        self.declare_parameter("state_hz", 1.0)

        self.port = self.get_parameter("port").get_parameter_value().string_value
        self.baud = int(self.get_parameter("baud").value)
        self.speed = int(self.get_parameter("speed").value)
        self.angle_count = int(self.get_parameter("angle_count").value)
        self.state_hz = max(0.2, float(self.get_parameter("state_hz").value))

        self.mc: Optional[MyCobot] = None
        self._connect()

        # PC에서 넘어오는 각도 명령을 구독
        self.command_sub = self.create_subscription(
            Float32MultiArray, "/arm/cmd", self._cmd_callback, 10
        )
        # 현재 관절 각도를 퍼블리시해서 모니터링에 활용
        self.state_pub = self.create_publisher(Float32MultiArray, "/arm/state", 10)

        self.timer = self.create_timer(1.0 / self.state_hz, self._publish_state)
        self.get_logger().info(
            f"✅ ArmFollowerNode ready (port={self.port}, speed={self.speed})"
        )

    def _connect(self) -> None:
        if self.mc is not None:
            return
        try:
            self.mc = MyCobot(self.port, self.baud)
            self.mc.power_on()
            self.get_logger().info("MyCobot connected.")
        except Exception as exc:
            self.mc = None
            self.get_logger().error(f"MyCobot 연결 실패: {exc}")

    def _cmd_callback(self, msg: Float32MultiArray) -> None:
        if self.mc is None:
            self._connect()
            if self.mc is None:
                self.get_logger().warn("MyCobot unavailable; dropping command.")
                return

        if len(msg.data) < self.angle_count:
            self.get_logger().warn(
                f"/arm/cmd expects {self.angle_count} angles, got {len(msg.data)}"
            )
            return

        angles = [float(v) for v in msg.data[: self.angle_count]]
        self._send_angles(angles)

    def _send_angles(self, angles: List[float]) -> None:
        if self.mc is None:
            return
        try:
            # 수신한 관절 벡터를 그대로 로봇에 전달
            self.mc.send_angles(angles, self.speed)
            self.get_logger().debug(f"sent angles {angles}")
        except Exception as exc:
            self.get_logger().error(f"send_angles 실패: {exc}")
            self.mc = None

    def _publish_state(self) -> None:
        if self.mc is None:
            self._connect()
            return
        try:
            current = self.mc.get_angles()
        except Exception as exc:
            self.get_logger().warn(f"현재 각도 조회 실패: {exc}")
            self.mc = None
            return
        if not current:
            return
        msg = Float32MultiArray()
        msg.data = [float(v) for v in current]
        self.state_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = ArmFollowerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
