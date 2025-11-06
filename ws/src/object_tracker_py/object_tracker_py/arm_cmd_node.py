import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class ArmCmdPub(Node):
    def __init__(self):
        super().__init__("arm_cmd_pub")
        self.pub = self.create_publisher(Float32MultiArray, "/arm/cmd", 10)

    def send(self, pan_cmd, tilt_cmd):
        msg = Float32MultiArray()
        msg.data = [float(pan_cmd), float(tilt_cmd)]
        self.pub.publish(msg)
        self.get_logger().info(f"팔 좌표 전송 pan={float(pan_cmd):+.4f}, tilt={float(tilt_cmd):+.4f}")


_arm_node = None


def init_ros2():
    global _arm_node
    rclpy.init()
    _arm_node = ArmCmdPub()


def spin_once():
    rclpy.spin_once(_arm_node, timeout_sec=0.0)


def send_cmd(pan_cmd, tilt_cmd):
    _arm_node.send(pan_cmd, tilt_cmd)
