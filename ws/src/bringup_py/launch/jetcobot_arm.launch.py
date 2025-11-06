from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    webcam_server = ExecuteProcess(
        cmd=["python3", "-m", "object_tracker_py.flask_webcam_server"],
        output="screen",
    )

    arm_follower = Node(
        package="tracking_control_py",
        executable="arm_follower_node",
        output="screen",
    )

    return LaunchDescription(
        [
            webcam_server,
            arm_follower,
        ]
    )
