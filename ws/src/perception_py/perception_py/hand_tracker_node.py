import cv2, numpy as np
import rclpy
from rclpy.node import Node
import mediapipe as mp
from geometry_msgs.msg import PointStamped


class HandTracker(Node):
    def __init__(self):
        super().__init__("hand_tracker")
        self.pub = self.create_publisher(PointStamped, "/target_point", 10)
        self.cap = cv2.VideoCapture(0)
        self.pose = mp.solutions.pose.Pose()

        self.timer = self.create_timer(1 / 30.0, self.loop)

    def loop(self):
        ok, frame = self.cap.read()
        if not ok:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if res.pose_landmarks:
            wrist = res.pose_landmarks.landmark[
                mp.solutions.pose.PoseLandmark.RIGHT_WRIST
            ]
            # 화면상 0~1 → 임시 mm 스케일(단안 가정; 1차 MVP)
            x_mm = (wrist.x - 0.5) * 300
            y_mm = (0.5 - wrist.y) * 300
            z_mm = 120.0
            msg = PointStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_frame"
            msg.point.x, msg.point.y, msg.point.z = x_mm, y_mm, z_mm
            self.pub.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(HandTracker())
    rclpy.shutdown()
