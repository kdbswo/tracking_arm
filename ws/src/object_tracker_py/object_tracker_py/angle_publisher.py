from __future__ import annotations

from typing import List

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class AnglePublisherNode(Node):
    """Skeleton node: publishes initial pose (angles) on /arm/cmd."""

    def __init__(self) -> None:
        super().__init__("angle_publisher_node")

        # 초기 포즈나 발행 방식은 파라미터로 간단히 바꿀 수 있도록 노출
        self.declare_parameter(
            "angles", [0.0, 0.0, 0.0, 0.0, 90.0, -45.0]
        )  # degrees
        self.declare_parameter("publish_once", True)
        self.declare_parameter("publish_hz", 1.0)

        angles_param = self.get_parameter("angles").value or []
        self.angles = [float(a) for a in angles_param] or [0.0] * 6
        self.publish_once = bool(self.get_parameter("publish_once").value)
        self.publish_hz = max(0.2, float(self.get_parameter("publish_hz").value))

        self.publisher = self.create_publisher(Float32MultiArray, "/arm/cmd", 10)

        if self.publish_once:
            self._publish_angles()
        else:
            self.timer = self.create_timer(1.0 / self.publish_hz, self._publish_angles)

        self.get_logger().info(
            f"AnglePublisherNode ready (publish_once={self.publish_once}, angles={self.angles})"
        )

    def _publish_angles(self) -> None:
        msg = Float32MultiArray()
        msg.data = self.angles
        self.publisher.publish(msg)
        self.get_logger().info("Published angles to /arm/cmd")


def main() -> None:
    rclpy.init()
    node = AnglePublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
