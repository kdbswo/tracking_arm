import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class ArmCmdPub(Node):
    def __init__(self):
        super().__init__("arm_cmd_pub")
        self.cmd_pub = self.create_publisher(Float32MultiArray, "/arm/cmd", 10)
        self.pose_pub = self.create_publisher(Float32MultiArray, "/arm/pose_cmd", 10)
        self.create_subscription(
            Float32MultiArray, "/arm/state", self._state_callback, 10
        )

    def send(self, pan_cmd, tilt_cmd):
        msg = Float32MultiArray()
        msg.data = [float(pan_cmd), float(tilt_cmd)]
        self.cmd_pub.publish(msg)
        self.get_logger().info(
            f"팔 좌표 전송 pan={float(pan_cmd):+.4f}, tilt={float(tilt_cmd):+.4f}"
        )

    def send_init_pose(self, pose_cmd):
        msg = Float32MultiArray()
        msg.data = pose_cmd
        self.pose_pub.publish(msg)
        self.get_logger().info(f"pose 자세 설정:{msg.data}")

    def _state_callback(self, msg):
        self.get_logger().info(f"현재 팔 각도: {msg.data}")


_arm_node = None


def init_ros2():
    global _arm_node
    rclpy.init()
    _arm_node = ArmCmdPub()


def spin_once():
    rclpy.spin_once(_arm_node, timeout_sec=0.0)


def send_cmd(pan_cmd, tilt_cmd):
    _arm_node.send(pan_cmd, tilt_cmd)


def send_init_pose(pose_cmd):
    _arm_node.send_init_pose(pose_cmd)

