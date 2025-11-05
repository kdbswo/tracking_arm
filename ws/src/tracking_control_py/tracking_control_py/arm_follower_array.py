# arm_follower_array.py
import os
import time
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from pymycobot import MyCobot


class ArmFollowerNode(Node):
    def __init__(self):
        super().__init__("arm_follower_node")

        # ---- 파라미터 ----
        self.declare_parameter("port", "/dev/ttyJETCOBOT")
        self.declare_parameter("baud", 115200)
        self.declare_parameter("speed", 30)
        # 팬/틸트에 대응할 조인트 인덱스 (예: J1=베이스 회전, J2=숄더 피치)
        self.declare_parameter("pan_joint", 0)   # J1
        self.declare_parameter("tilt_joint", 1)  # J2
        # 제어 주기(Hz), 각속도 스케일, 각도 제한
        self.declare_parameter("control_hz", 30.0)
        self.declare_parameter("cmd_scale", 1.0)         # YOLO -> 각속도(rad/s) 스케일
        self.declare_parameter("deg_limit", 80.0)        # 각도 제한(절대값, 도)
        self.declare_parameter("deadband", 0.002)        # 너무 작은 명령 무시(라디안/초)

        self.port = self.get_parameter("port").value
        self.baud = int(self.get_parameter("baud").value)
        self.speed = int(self.get_parameter("speed").value)
        self.pan_joint = int(self.get_parameter("pan_joint").value)
        self.tilt_joint = int(self.get_parameter("tilt_joint").value)
        self.control_hz = float(self.get_parameter("control_hz").value)
        self.cmd_scale = float(self.get_parameter("cmd_scale").value)
        self.deg_limit = float(self.get_parameter("deg_limit").value)
        self.deadband = float(self.get_parameter("deadband").value)

        # ---- 하드웨어 연결 ----
        self.mc = None
        self._connect()

        # 현재 목표 각도(도) – 6축 기준
        self.target_deg = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # 마지막으로 받은 명령(라디안/초 가정)
        self.last_pan_vel = 0.0
        self.last_tilt_vel = 0.0

        # 구독자: YOLO 쪽에서 퍼블리시하는 [pan_cmd, tilt_cmd]
        self.sub = self.create_subscription(
            Float32MultiArray, "/arm/cmd", self.cmd_callback, 10
        )

        # 제어 루프 타이머
        self.dt = 1.0 / self.control_hz
        self.timer = self.create_timer(self.dt, self._control_step)

        self.get_logger().info("✅ ArmFollowerNode started. Subscribing /arm/cmd")

    def _connect(self):
        if self.mc is not None:
            return
        try:
            self.mc = MyCobot(self.port, self.baud)
            self.get_logger().info(f"MyCobot 연결 완료: {self.port} @ {self.baud}bps")
        except Exception as exc:
            self.mc = None
            self.get_logger().warn(f"MyCobot 연결 실패: {exc}")

    def cmd_callback(self, msg: Float32MultiArray):
        if len(msg.data) < 2:
            self.get_logger().warn("잘못된 cmd 데이터. [pan, tilt] 필요")
            return

        pan_cmd = float(msg.data[0])  # 보통 라디안/초
        tilt_cmd = float(msg.data[1])

        # 너무 작은 값은 무시(잡음 제거)
        if abs(pan_cmd) < self.deadband:
            pan_cmd = 0.0
        if abs(tilt_cmd) < self.deadband:
            tilt_cmd = 0.0

        # 스케일 적용
        self.last_pan_vel = pan_cmd * self.cmd_scale
        self.last_tilt_vel = tilt_cmd * self.cmd_scale

    def _control_step(self):
        if self.mc is None:
            return

        # 라디안/초 명령 → dt 적분 → 라디안 → 도 변환
        d_pan_rad = self.last_pan_vel * self.dt
        d_tilt_rad = self.last_tilt_vel * self.dt

        d_pan_deg = np.degrees(d_pan_rad)
        d_tilt_deg = np.degrees(d_tilt_rad)

        # 목표 각도 갱신
        self.target_deg[self.pan_joint]  += d_pan_deg
        self.target_deg[self.tilt_joint] += d_tilt_deg

        # 안전 제한
        self.target_deg[self.pan_joint]  = float(np.clip(self.target_deg[self.pan_joint],  -self.deg_limit, self.deg_limit))
        self.target_deg[self.tilt_joint] = float(np.clip(self.target_deg[self.tilt_joint], -self.deg_limit, self.deg_limit))

        try:
            self.mc.send_angles(self.target_deg, self.speed)
        except Exception as exc:
            self.get_logger().warn(f"각도 전송 실패: {exc}")
