from setuptools import find_packages, setup
import os
from glob import glob # 用来匹配文件路径模式

package_name = 'volcano_ark_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # --- 这是关键的一行 (或几行) ---
        # 安装 launch 目录下的所有 .launch.py 文件
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
        # 如果你有其他 .launch 文件类型 (如 .xml, .yaml)，也需要类似地添加：
        # (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.xml'))),
        # (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.yaml'))),

        # 如果你有配置文件，也需要类似地安装：
        # (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'web_interface_node = volcano_ark_ros.web_interface_node:main',
            'originman_action_node = volcano_ark_ros.originman_action_node:main',
            'image_caption_node = volcano_ark_ros.image_caption_node:main',
            'image_publisher_node = volcano_ark_ros.image_publisher_node:main',
            'text_to_speech_node = volcano_ark_ros.text_to_speech_node:main',
        ],
    },
)
