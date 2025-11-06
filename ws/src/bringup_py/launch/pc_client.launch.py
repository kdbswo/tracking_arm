from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description():
    app_process = ExecuteProcess(
        cmd=["python3", "-m", "object_tracker_py.app"],
        output="screen",
    )

    viewer_process = ExecuteProcess(
        cmd=["python3", "-m", "object_tracker_py.flask_webcam_client"],
        output="screen",
    )

    return LaunchDescription(
        [
            app_process,
            viewer_process,
        ]
    )
