from setuptools import find_packages, setup

package_name = "tracking_control_py"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy", "pymycobot", "rclpy", "std_msgs"],
    zip_safe=True,
    maintainer="addinedu",
    maintainer_email="kdbswo@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "motion_test_node = tracking_control_py.motion_test_node:main",
            "arm_follower_node = tracking_control_py.arm_follower_node:main",
        ],
    },
)
