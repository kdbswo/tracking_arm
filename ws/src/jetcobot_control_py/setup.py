from setuptools import find_packages, setup

package_name = "jetcobot_control_py"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "opencv-python", "pymycobot"],
    zip_safe=True,
    maintainer="addinedu",
    maintainer_email="kdbswo@gmail.com",
    description="Minimal MyCobot control node with UDP video streaming.",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "jetcobot_udp_node = jetcobot_control_py.jet_cobot_udp_node:main",
        ],
    },
)
