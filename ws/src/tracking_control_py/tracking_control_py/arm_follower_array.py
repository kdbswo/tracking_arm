# arm_follower_array.py
from __future__ import annotations

import math
from typing import Optional

import rclpy
from pymycobot import MyCobot
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class ArmFollowerNode(Node):
    """MyCobot visual-servo controller (Option A: joint5 yaw + Cartesian axis)."""

    def __init__(self) -> None:
        super().__init__("arm_follower_node")

        # Connection / timing
        self.declare_parameter("port", "/dev/ttyJETCOBOT")
        self.declare_parameter("baud", 1_000_000)
        self.declare_parameter("speed", 40)
        self.declare_parameter("control_hz", 15.0)
        self.declare_parameter("target_timeout", 0.6)

        # Horizontal yaw (joint 5) config
        self.declare_parameter("yaw_joint", 4)
        self.declare_parameter("yaw_min_deg", -120.0)
        self.declare_parameter("yaw_max_deg", 120.0)
        self.declare_parameter("kp_x", 0.0)  # if zero, auto from FOV + alpha
        self.declare_parameter("fov_horizontal_deg", 70.0)
        self.declare_parameter("yaw_gain_alpha", 0.6)
        self.declare_parameter("max_yaw_step_deg", 2.0)
        self.declare_parameter("deadband", 0.03)
        self.declare_parameter("flip_x", False)

        # Vertical (Option A: Cartesian axis) config
        self.declare_parameter("enable_vertical", True)
        self.declare_parameter("cartesian_axis", "z")  # 'z' or 'y'
        self.declare_parameter("kp_y", 1.0)
        self.declare_parameter("z_scale_mm", 10.0)
        self.declare_parameter("max_z_step_mm", 5.0)
        self.declare_parameter("axis_min_mm", 80.0)
        self.declare_parameter("axis_max_mm", 320.0)
        self.declare_parameter("flip_y", False)
        self.declare_parameter("coord_mode", 1)

        self.declare_parameter("angle_refresh_sec", 4.0)
        self.declare_parameter("restore_orientation", True)
        self.declare_parameter("orientation_tolerance_deg", 1.0)

        # Load parameters
        self.port = self.get_parameter("port").get_parameter_value().string_value
        self.baud = int(self.get_parameter("baud").value)
        self.speed = int(self.get_parameter("speed").value)
        self.control_hz = max(1.0, float(self.get_parameter("control_hz").value))
        self.target_timeout = float(self.get_parameter("target_timeout").value)

        self.yaw_joint = int(self.get_parameter("yaw_joint").value)
        self.yaw_min_deg = float(self.get_parameter("yaw_min_deg").value)
        self.yaw_max_deg = float(self.get_parameter("yaw_max_deg").value)
        self.kp_x = float(self.get_parameter("kp_x").value)
        self.fov_horizontal_deg = float(self.get_parameter("fov_horizontal_deg").value)
        self.yaw_gain_alpha = float(self.get_parameter("yaw_gain_alpha").value)
        self.max_yaw_step_deg = float(self.get_parameter("max_yaw_step_deg").value)
        self.deadband = float(self.get_parameter("deadband").value)
        self.flip_x = bool(self.get_parameter("flip_x").value)

        self.enable_vertical = bool(self.get_parameter("enable_vertical").value)
        self.cartesian_axis = (
            self.get_parameter("cartesian_axis").get_parameter_value().string_value
        ).lower()
        self.kp_y = float(self.get_parameter("kp_y").value)
        self.z_scale_mm = float(self.get_parameter("z_scale_mm").value)
        self.max_z_step_mm = float(self.get_parameter("max_z_step_mm").value)
        self.axis_min_mm = float(self.get_parameter("axis_min_mm").value)
        self.axis_max_mm = float(self.get_parameter("axis_max_mm").value)
        self.flip_y = bool(self.get_parameter("flip_y").value)
        self.coord_mode = int(self.get_parameter("coord_mode").value)

        self.angle_refresh_sec = float(self.get_parameter("angle_refresh_sec").value)
        self.restore_orientation = bool(self.get_parameter("restore_orientation").value)
        self.orientation_tolerance_deg = float(
            self.get_parameter("orientation_tolerance_deg").value
        )

        if self.kp_x <= 0.0:
            self.kp_x = math.radians(self.fov_horizontal_deg / 2.0) * self.yaw_gain_alpha
            self.get_logger().info(
                f"Auto kp_x set to {self.kp_x:.3f} rad/e (FOV={self.fov_horizontal_deg}°, alpha={self.yaw_gain_alpha})"
            )

        self.axis_index = 2 if self.cartesian_axis == "z" else 1

        # Robot connection and state
        self.mc: Optional[MyCobot] = None
        self.target_deg = [0.0] * 6
        self._last_pose: Optional[list[float]] = None
        self._connect()

        # Command tracking
        self.last_ex = 0.0
        self.last_ey = 0.0
        self._target_active = False
        self._last_cmd_time = self.get_clock().now()
        self._last_angle_refresh = self.get_clock().now()

        # ROS interfaces
        self.create_subscription(Float32MultiArray, "/arm/cmd", self.cmd_callback, 10)
        self.pose_sub = self.create_subscription(
            Float32MultiArray, "/arm/pose_cmd", self.pose_cmd_callback, 10
        )
        self.state_pub = self.create_publisher(Float32MultiArray, "/arm/state", 10)

        self.timer = self.create_timer(1.0 / self.control_hz, self._control_step)
        self.get_logger().info("✅ ArmFollowerNode ready (Option A yaw+Z control)")

    def _connect(self) -> None:
        if self.mc is not None:
            return
        try:
            self.mc = MyCobot(self.port, self.baud)
            self.get_logger().info(f"MyCobot 연결 완료: {self.port} @ {self.baud}bps")
            angles = self.mc.get_angles()
            if angles:
                self.target_deg = [float(a) for a in angles]
            pose = self._safe_get_coords()
            if pose:
                self._last_pose = pose[:]
        except Exception as exc:  # pragma: no cover - hardware specific
            self.mc = None
            self.get_logger().warn(f"MyCobot 연결 실패: {exc}")

    def cmd_callback(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 2:
            self.get_logger().warn("잘못된 cmd 데이터. [ex, ey] 필요")
            return
        self.last_ex = float(msg.data[0])
        self.last_ey = float(msg.data[1])
        self._target_active = True
        self._last_cmd_time = self.get_clock().now()

    def pose_cmd_callback(self, msg: Float32MultiArray) -> None:
        if self.mc is None:
            return
        if len(msg.data) != 6:
            self.get_logger().warn("초기 자세는 6개의 각도가 있어야 합니다.")
            return
        angles = [float(v) for v in msg.data]
        try:
            self.mc.send_angles(angles, self.speed)
            self.target_deg = angles[:]
        except Exception as exc:  # pragma: no cover
            self.get_logger().warn(f"초기 자세 전송 실패: {exc}")

    def _control_step(self) -> None:
        if self.mc is None:
            self._connect()
            return

        now = self.get_clock().now()
        age = (now - self._last_cmd_time).nanoseconds * 1e-9
        if not self._target_active or age > self.target_timeout:
            self._target_active = False
            self._publish_state()
            return

        ex = -self.last_ex if self.flip_x else self.last_ex
        ey = -self.last_ey if self.flip_y else self.last_ey

        moved = False

        if abs(ex) > self.deadband:
            moved = self._apply_yaw(ex) or moved

        if self.enable_vertical and abs(ey) > self.deadband:
            moved = self._apply_vertical(ey) or moved

        if moved:
            self._maybe_refresh_angles(now)

        self._publish_state()

    def _apply_yaw(self, ex: float) -> bool:
        if self.mc is None:
            return False
        delta_rad = self.kp_x * ex
        delta_deg = clamp(
            math.degrees(delta_rad), -self.max_yaw_step_deg, self.max_yaw_step_deg
        )
        if abs(delta_deg) < 1e-3:
            return False

        current = self.target_deg[self.yaw_joint] + delta_deg
        current = clamp(current, self.yaw_min_deg, self.yaw_max_deg)
        self.target_deg[self.yaw_joint] = current

        try:
            yaw_joint_id = self.yaw_joint + 1  # pymycobot joints are 1-based
            self.mc.send_angle(yaw_joint_id, current, self.speed)
            return True
        except Exception as exc:  # pragma: no cover
            self.get_logger().warn(f"Yaw command failed: {exc}")
            return False

    def _apply_vertical(self, ey: float) -> bool:
        if self.mc is None:
            return False
        delta_mm = clamp(self.kp_y * ey * self.z_scale_mm, -self.max_z_step_mm, self.max_z_step_mm)
        if abs(delta_mm) < 1e-3:
            return False

        coords = self._safe_get_coords()
        if not coords or len(coords) < 6:
            return False

        new_axis = clamp(
            coords[self.axis_index] + delta_mm, self.axis_min_mm, self.axis_max_mm
        )
        axis_id = self.axis_index + 1  # pymycobot axis id is 1-based

        try:
            self.mc.send_coord(axis_id, new_axis, self.speed)
            if self.restore_orientation:
                self._restore_orientation(coords)
            return True
        except Exception as exc:  # pragma: no cover
            self.get_logger().warn(f"Axis move failed: {exc}")
            return False

    def _maybe_refresh_angles(self, now) -> None:
        if (now - self._last_angle_refresh).nanoseconds * 1e-9 < self.angle_refresh_sec:
            return
        if self.mc is None:
            return
        try:
            current = self.mc.get_angles()
            if current:
                self.target_deg = [float(a) for a in current]
                self._last_angle_refresh = now
        except Exception:
            pass

    def _safe_get_coords(self) -> Optional[list[float]]:
        if self.mc is None:
            return None
        try:
            coords = self.mc.get_coords()
        except Exception as exc:
            self.get_logger().warn(f"좌표 조회 실패: {exc}")
            return None
        if coords:
            self._last_pose = coords[:]
        return coords

    def _restore_orientation(self, reference_pose: list[float]) -> None:
        if not self.restore_orientation or self.mc is None or not reference_pose:
            return
        try:
            current = self.mc.get_coords()
        except Exception:
            return
        if not current or len(current) < 6:
            return

        diffs = [
            abs(current[3] - reference_pose[3]),
            abs(current[4] - reference_pose[4]),
            abs(current[5] - reference_pose[5]),
        ]
        if max(diffs) < self.orientation_tolerance_deg:
            self._last_pose = current[:]
            return

        target = current[:]
        target[3] = reference_pose[3]
        target[4] = reference_pose[4]
        target[5] = reference_pose[5]
        try:
            self.mc.send_coords(target, self.speed, self.coord_mode)
            self._last_pose = target
        except Exception as exc:
            self.get_logger().warn(f"Orientation restore failed: {exc}")

    def _publish_state(self) -> None:
        if self.mc is None:
            return
        try:
            current_deg = self.mc.get_angles()
        except Exception as exc:
            self.get_logger().warn(f"현재 각도 조회 실패: {exc}")
            return
        if not current_deg:
            return
        msg = Float32MultiArray()
        msg.data = [float(angle) for angle in current_deg]
        self.state_pub.publish(msg)


def main():
    rclpy.init()
    node = ArmFollowerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
