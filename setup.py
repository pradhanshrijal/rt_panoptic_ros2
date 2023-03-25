from setuptools import setup

package_name = 'rt_panoptic_ros2'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
        ],
    },
)
