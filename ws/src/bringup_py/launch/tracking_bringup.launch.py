from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="tracking_control_py",
                executable="motion_test_node",
                name="motion_test",
            ),
        ]
    )
