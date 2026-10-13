from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'x2_recovery_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        (
            os.path.join('share', package_name),
            ['package.xml'],
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='islam',
    maintainer_email='islam.salih@smccl.co.uk',
    description='ROS 2 package for AgiBot X2 ground recovery',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'recovery_node = x2_recovery_ros.recovery_node:main',
            'telemetry_node = x2_recovery_ros.telemetry_node:main',
        ],
    },
)