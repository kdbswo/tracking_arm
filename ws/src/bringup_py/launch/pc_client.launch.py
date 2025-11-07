import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    bridge_config = os.path.join(
        get_package_share_directory("bringup_py"),
        "config",
        "domain_bridge_arm_cmd.yaml",
    )

    app_process = ExecuteProcess(
        cmd=["python3", "-m", "object_tracker_py.app"],
        output="screen",
    )

    # viewer_process = ExecuteProcess(
    #     cmd=["python3", "-m", "object_tracker_py.flask_webcam_client"],
    #     output="screen",
    # )

    domain_bridge_node = Node(
        package="domain_bridge",
        executable="domain_bridge",
        name="arm_cmd_bridge",
        arguments=[bridge_config],
        output="screen",
    )

    return LaunchDescription(
        [
            app_process,
            # viewer_process,
            domain_bridge_node,
        ]
    )
