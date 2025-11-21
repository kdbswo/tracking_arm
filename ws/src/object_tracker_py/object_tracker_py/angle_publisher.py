from __future__ import annotations

import time
from typing import List

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class AnglePublisherNode(Node):
    """Skeleton node: publishes initial pose (angles) on /arm/cmd."""

    def __init__(self) -> None:
        super().__init__("angle_publisher_node")

        # 초기 포즈나 발행 방식은 파라미터로 간단히 바꿀 수 있도록 노출
        self.declare_parameter("angles", [0.0, 0.0, 0.0, 0.0, 90.0, 0.0])  # degrees
        self.declare_parameter("publish_once", True)
        self.declare_parameter("publish_hz", 1.0)
        self.declare_parameter("pose_publish_delay_sec", 2.0)

        angles_param = self.get_parameter("angles").value or []
        self.angles = [float(a) for a in angles_param] or [0.0] * 6
        self.publish_once = bool(self.get_parameter("publish_once").value)
        self.publish_hz = max(0.2, float(self.get_parameter("publish_hz").value))
        self.publish_delay_sec = max(
            0.0, float(self.get_parameter("pose_publish_delay_sec").value)
        )

        self.publisher = self.create_publisher(Float32MultiArray, "/arm/cmd", 10)
        self.pose_publisher = self.create_publisher(
            Float32MultiArray, "/arm/pose_cmd", 10
        )
        self.create_subscription(
            Float32MultiArray, "/arm/state", self._state_callback, 10
        )
        self._last_state_log = 0.0

        self._one_shot_timer = None

        if self.publish_once:
            if self.publish_delay_sec > 0.0:
                self._one_shot_timer = self.create_timer(
                    self.publish_delay_sec, self._publish_once_timer_cb
                )
            else:
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
        pose_msg = Float32MultiArray()
        pose_msg.data = self.angles
        self.pose_publisher.publish(pose_msg)
        self.get_logger().info("Published angles to /arm/cmd and /arm/pose_cmd")

    def _publish_once_timer_cb(self) -> None:
        if self._one_shot_timer is not None:
            self._one_shot_timer.cancel()
            self._one_shot_timer = None
        self._publish_angles()

    def _state_callback(self, msg: Float32MultiArray) -> None:
        now = time.monotonic()
        if now - self._last_state_log < 5.0:
            return
        self._last_state_log = now
        self.get_logger().info(f"현재 팔 각도: {msg.data}")


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
