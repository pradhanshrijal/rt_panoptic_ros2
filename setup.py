from setuptools import setup
import glob
import os

package_name = 'rt_panoptic_ros2'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
            glob.glob(os.path.join('launch', '*'))),
        ('share/' + package_name + '/config',
            glob.glob(os.path.join('config', '*'))),
        ('share/' + package_name + '/cfgs',
            glob.glob(os.path.join('cfgs', '*.*'))),
        ('share/' + package_name + '/checkpoints',
            glob.glob(os.path.join('checkpoints', '*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Shrijal Pradhan',
    maintainer_email='pradhan.shrijal@gmail.com',
    description='ROS 2 Wrapper for realtime_panoptic',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'panoptic = rt_panoptic_ros2.panoptic_node:main',
        ],
    },
)
