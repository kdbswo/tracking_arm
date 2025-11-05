import rclpy

from tracking_control_py.arm_follower_array import ArmFollowerNode


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
