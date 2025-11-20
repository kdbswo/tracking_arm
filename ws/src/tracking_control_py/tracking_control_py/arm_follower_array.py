# arm_follower_array.py
from __future__ import annotations

import math
from typing import Optional
import time

import rclpy
from pymycobot import MyCobot
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class ArmFollowerNode(Node):
    """MyCobot visual-servo controller (yaw only on joint 5)."""

    def __init__(self) -> None:
        super().__init__("arm_follower_node")

        # Connection / timing
        self.declare_parameter("port", "/dev/ttyJETCOBOT")
        self.declare_parameter("baud", 1_000_000)
        self.declare_parameter("speed", 40)
        self.declare_parameter("control_hz", 15.0)
        self.declare_parameter("target_timeout", 0.1)
        self.declare_parameter("center_enter_th", 0.1)
        self.declare_parameter("center_exit_th", 0.15)

        # Horizontal yaw (joint 5) config
        self.declare_parameter("yaw_joint", 4)
        self.declare_parameter("yaw_min_deg", -120.0)
        self.declare_parameter("yaw_max_deg", 120.0)
        self.declare_parameter("kp_x", 0.0)  # if zero, auto from FOV + alpha
        self.declare_parameter("fov_horizontal_deg", 70.0)
        self.declare_parameter("yaw_gain_alpha", 0.6)
        self.declare_parameter("max_yaw_step_deg", 2.0)
        self.declare_parameter("deadband", 0.03)
        self.declare_parameter("flip_x", True)
        self.declare_parameter("yaw_sign", 1)
        self.declare_parameter(
            "yaw_control_mode", "velocity"
        )  # "position" or "velocity"
        # 속도 제어: 오차와 무관하게 일정 속도로 조그
        self.declare_parameter("yaw_vel_const", 10.0)
        self.declare_parameter("yaw_deadband", 0.03)
        self.declare_parameter("yaw_stick_ms", 0)
        self.declare_parameter("yaw_dir_pos", 0)
        self.declare_parameter("yaw_dir_neg", 1)

        self.declare_parameter("angle_refresh_sec", 4.0)
        self.declare_parameter("state_publish_sec", 2.0)

        # Load parameters
        self.port = self.get_parameter("port").get_parameter_value().string_value
        self.baud = int(self.get_parameter("baud").value)
        self.speed = int(self.get_parameter("speed").value)
        self.control_hz = max(1.0, float(self.get_parameter("control_hz").value))
        self.target_timeout = float(self.get_parameter("target_timeout").value)
        self.center_enter_th = float(self.get_parameter("center_enter_th").value)
        self.center_exit_th = float(self.get_parameter("center_exit_th").value)

        self.yaw_joint = int(self.get_parameter("yaw_joint").value)
        self.yaw_min_deg = float(self.get_parameter("yaw_min_deg").value)
        self.yaw_max_deg = float(self.get_parameter("yaw_max_deg").value)
        self.kp_x = float(self.get_parameter("kp_x").value)
        self.fov_horizontal_deg = float(self.get_parameter("fov_horizontal_deg").value)
        self.yaw_gain_alpha = float(self.get_parameter("yaw_gain_alpha").value)
        self.max_yaw_step_deg = float(self.get_parameter("max_yaw_step_deg").value)
        self.deadband = float(self.get_parameter("deadband").value)
        self.flip_x = bool(self.get_parameter("flip_x").value)
        self.yaw_sign = int(self.get_parameter("yaw_sign").value) or 1
        self.yaw_control_mode = (
            self.get_parameter("yaw_control_mode")
            .get_parameter_value()
            .string_value.lower()
        )
        self.yaw_vel_const = float(self.get_parameter("yaw_vel_const").value)
        self.yaw_deadband = float(self.get_parameter("yaw_deadband").value)
        self.yaw_stick_ms = int(self.get_parameter("yaw_stick_ms").value)
        self.yaw_dir_pos = int(self.get_parameter("yaw_dir_pos").value)
        self.yaw_dir_neg = int(self.get_parameter("yaw_dir_neg").value)

        self.angle_refresh_sec = float(self.get_parameter("angle_refresh_sec").value)
        self.state_publish_sec = float(self.get_parameter("state_publish_sec").value)

        if self.kp_x <= 0.0:
            self.kp_x = (
                math.radians(self.fov_horizontal_deg / 2.0) * self.yaw_gain_alpha
            )
            self.get_logger().info(
                f"Auto kp_x set to {self.kp_x:.3f} rad/e (FOV={self.fov_horizontal_deg}°, alpha={self.yaw_gain_alpha})"
            )

        # Robot connection and state
        self.mc: Optional[MyCobot] = None
        self.target_deg = [0.0] * 6
        self._connect()

        # Command tracking
        self.last_ex = 0.0
        self._target_px = None
        self._frame_center = None
        self._center_state = False
        self._target_active = False
        self._last_cmd_time = None
        self._last_px_time = None
        self._last_angle_refresh = self.get_clock().now()
        self._last_state_pub_time = None
        self._yaw_jog_dir = 0  # -1, 0, +1 (velocity jog state)
        self._yaw_last_active = self.get_clock().now()
        self._last_target_log_sec = 0.0

        # ROS interfaces
        self.create_subscription(Float32MultiArray, "/arm/cmd", self.cmd_callback, 10)
        self.create_subscription(
            Float32MultiArray, "/arm/target_px", self.target_px_callback, 10
        )
        self.pose_sub = self.create_subscription(
            Float32MultiArray, "/arm/pose_cmd", self.pose_cmd_callback, 10
        )
        self.state_pub = self.create_publisher(Float32MultiArray, "/arm/state", 10)

        self.timer = self.create_timer(1.0 / self.control_hz, self._control_step)
        self.get_logger().info(
            f"✅ ArmFollowerNode ready (yaw-only control) "
            f"[flip_x={self.flip_x}, yaw_sign={self.yaw_sign}, yaw_joint={self.yaw_joint + 1}]"
        )

    def _connect(self) -> None:
        if self.mc is not None:
            return
        try:
            self.mc = MyCobot(self.port, self.baud)
            self.get_logger().info(f"MyCobot 연결 완료: {self.port} @ {self.baud}bps")
            angles = self.mc.get_angles()
            if angles:
                self.target_deg = [float(a) for a in angles]
        except Exception as exc:  # pragma: no cover - hardware specific
            self.mc = None
            self.get_logger().warn(f"MyCobot 연결 실패: {exc}")

    def cmd_callback(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 2:
            self.get_logger().warn("잘못된 cmd 데이터. [ex, ey] 필요")
            return
        self.last_ex = float(msg.data[0])
        self._target_active = True
        self._last_cmd_time = self.get_clock().now()
        self.get_logger().info(f"/arm/cmd 수신 ex={self.last_ex:+.3f}")

    def target_px_callback(self, msg: Float32MultiArray) -> None:
        now = self.get_clock().now()
        if len(msg.data) < 4:
            # 명시적으로 타겟 해제
            self._target_px = None
            self._frame_center = None
            self._center_state = False
            self._target_active = False
            self._last_px_time = now
            if self.yaw_control_mode == "velocity":
                self._yaw_stop()
            self.get_logger().info("/arm/target_px 해제 수신 -> 정지")
            return

        cx, cy, w, h = [float(v) for v in msg.data[:4]]
        if w <= 0.0 or h <= 0.0:
            self.get_logger().warn("/arm/target_px 데이터가 잘못됨 (w/h<=0)")
            return
        self._target_px = (cx, cy)
        self._frame_center = (w / 2.0, h / 2.0)
        self._target_active = True
        self._last_px_time = now
        self._center_state = False
        now_sec = now.nanoseconds * 1e-9
        if now_sec - self._last_target_log_sec > 1.0:
            self.get_logger().info(
                f"/arm/target_px 수신: cx={cx:.1f}, cy={cy:.1f}, w={w:.0f}, h={h:.0f}"
            )
            self._last_target_log_sec = now_sec

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
        ex = None

        if self._last_px_time is not None:
            age_px = (now - self._last_px_time).nanoseconds * 1e-9
            if age_px <= self.target_timeout:
                ex = self._calc_ex_from_px()

        if ex is None and self._last_cmd_time is not None:
            age_cmd = (now - self._last_cmd_time).nanoseconds * 1e-9
            if self._target_active and age_cmd <= self.target_timeout:
                ex = -self.last_ex if self.flip_x else self.last_ex

        if ex is None:
            self._target_active = False
            if self.yaw_control_mode == "velocity":
                self._yaw_stop()
                self.get_logger().info("타임아웃/타겟 없음으로 yaw 정지")
            self._publish_state()
            return

        self._target_active = True

        moved = False

        if self.yaw_control_mode == "velocity":
            moved = self._apply_yaw_velocity(ex) or moved
        elif abs(ex) > self.deadband:
            moved = self._apply_yaw(ex) or moved

        if moved:
            self._maybe_refresh_angles(now)

        self._publish_state()

    def _calc_ex_from_px(self) -> Optional[float]:
        """Convert target pixel + frame center into normalized ex with hysteresis."""
        if self._target_px is None or self._frame_center is None:
            return None
        tx, _ty = self._target_px
        cx, _cy = self._frame_center
        if cx == 0.0:
            return None
        ex = clamp((tx - cx) / cx, -1.0, 1.0)

        if self._center_state:
            if abs(ex) <= self.center_exit_th:
                ex = 0.0
            else:
                self._center_state = False
        else:
            if abs(ex) <= self.center_enter_th:
                self._center_state = True
                ex = 0.0

        return -ex if self.flip_x else ex

    def _apply_yaw(self, ex: float) -> bool:
        if self.mc is None:
            return False
        delta_rad = self.yaw_sign * self.kp_x * ex
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

    def _apply_yaw_velocity(self, ex: float) -> bool:
        """Velocity-style yaw jog control fed by normalized ex."""
        if self.mc is None:
            return False

        signed_ex = self.yaw_sign * ex
        now = self.get_clock().now()

        if abs(signed_ex) <= self.yaw_deadband:
            # 즉시 정지: 데드밴드 안쪽이면 바로 jog_stop 호출
            if self._yaw_jog_dir != 0:
                self._yaw_stop()
                self.get_logger().info("데드밴드로 yaw 정지(즉시)")
            return False

        self._yaw_last_active = now

        # 오차 크기와 무관하게 일정 속도로 구동
        spd = int(self.yaw_vel_const)
        direction = 1 if signed_ex > 0 else -1

        if direction != self._yaw_jog_dir:
            if self._yaw_jog_dir != 0:
                self._yaw_stop()
            self._yaw_start(direction, spd)
            return True

        self._yaw_speed(spd)
        return True

    def _yaw_start(self, direction: int, speed: int) -> None:
        if self.mc is None:
            return
        joint_id = self.yaw_joint + 1
        dir_code = self.yaw_dir_pos if direction > 0 else self.yaw_dir_neg
        try:
            if hasattr(self.mc, "jog_angle"):
                self.mc.jog_angle(joint_id, dir_code, int(speed))
            else:
                delta = 1.0 if direction > 0 else -1.0
                angles = self.target_deg[:]
                angles[self.yaw_joint] = angles[self.yaw_joint] + delta
                self.mc.send_angles(angles, int(speed))
                self.target_deg = angles
            self._yaw_jog_dir = direction
        except Exception as exc:
            self.get_logger().warn(f"Yaw jog start failed: {exc}")
            self._yaw_jog_dir = 0
        else:
            self.get_logger().info(f"Yaw 시작 dir={direction:+d} speed={speed}")

    def _yaw_speed(self, speed: int) -> None:
        if self.mc is None or self._yaw_jog_dir == 0:
            return
        joint_id = self.yaw_joint + 1
        dir_code = self.yaw_dir_pos if self._yaw_jog_dir > 0 else self.yaw_dir_neg
        try:
            if hasattr(self.mc, "jog_angle"):
                self.mc.jog_angle(joint_id, dir_code, int(speed))
        except Exception as exc:
            self.get_logger().warn(f"Yaw jog speed update failed: {exc}")

    def _yaw_stop(self) -> None:
        if self.mc is None or self._yaw_jog_dir == 0:
            return
        try:
            if hasattr(self.mc, "jog_stop"):
                self.mc.jog_stop()
            elif hasattr(self.mc, "stop"):
                self.mc.stop()
        except Exception as exc:
            self.get_logger().warn(f"Yaw jog stop failed: {exc}")
        finally:
            self._yaw_jog_dir = 0
            self.get_logger().info("Yaw 정지 완료")

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

    def _publish_state(self) -> None:
        if self.mc is None:
            return
        now = self.get_clock().now()
        if self._last_state_pub_time is not None:
            age = (now - self._last_state_pub_time).nanoseconds * 1e-9
            if age < self.state_publish_sec:
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
        self._last_state_pub_time = now


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
