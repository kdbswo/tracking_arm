from setuptools import find_packages, setup

package_name = 'bringup_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (
            'share/' + package_name + '/config',
            ['config/domain_bridge_arm_cmd.yaml'],
        ),
        (
            'share/' + package_name + '/launch',
            [
                'launch/tracking_bringup.launch.py',
                'launch/pc_client.launch.py',
                'launch/jetcobot_arm.launch.py',
            ],
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='addinedu',
    maintainer_email='kdbswo@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
