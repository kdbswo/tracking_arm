from setuptools import find_packages, setup

package_name = "object_tracker_py"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=[
        "setuptools",
        "Flask",
        "opencv-python",
        "numpy",
        "requests",
        "ultralytics",
        "torch",
        "rclpy",
        "std_msgs",
    ],
    zip_safe=True,
    maintainer="addinedu",
    maintainer_email="kdbswo@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "pc_tracker_app = object_tracker_py.app:main",
            "angle_cmd_publisher = object_tracker_py.angle_publisher:main",
            "flask_webcam_server = object_tracker_py.flask_webcam_server:main",
            "flask_webcam_client = object_tracker_py.flask_webcam_client:main",
        ]
    },
)
