from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="perception_py",
                executable="hand_tracker_node",
                name="hand_tracker",
            ),
            Node(
                package="tracking_control_py",
                executable="follow_node",
                name="follow_node",
            ),
        ]
    )
