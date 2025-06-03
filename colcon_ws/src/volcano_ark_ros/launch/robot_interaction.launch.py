# volcano_ark_ros/launch/robot_interaction.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    your_pkg_name = 'volcano_ark_ros' # <--- **修改为你的包名**

    web_interface_node = Node(
        package=your_pkg_name,
        executable='web_interface_node',
        name='web_interface_node',
        output='screen'
    )

    originman_action_node = Node(
        package=your_pkg_name,
        executable='originman_action_node',
        name='originman_action_node',
        output='screen'
    )

    # image_publisher_node 带参数启动
    image_publisher_node = Node(
        package=your_pkg_name,
        executable='image_publisher_node',
        name='image_publisher_node',
        output='screen',
        parameters=[
            {'camera_index': 0},         # 摄像头索引
            {'publish_frequency': 2.0}   # 发布频率 (Hz)
        ]
    )

    image_caption_node = Node(
        package=your_pkg_name,
        executable='image_caption_node',
        name='image_caption_node',
        output='screen'
    )

    # text_to_speech_node 带参数启动
    text_to_speech_node = Node(
        package=your_pkg_name,
        executable='text_to_speech_node',
        name='text_to_speech_node',
        output='screen',
        parameters=[
            {'audio_device': 'default'} # 或者 'plughw:0,0', 'hw:0,0' 等
        ]
    )

    rosbridge_server_node = Node(
        package='rosbridge_server',
        executable='rosbridge_websocket',
        name='rosbridge_websocket',
        parameters=[{'port': 9090}] # 默认端口是9090，可以按需修改
    )
    
    # (可选) 启动 rosbridge_server
    rosbridge_server_pkg = get_package_share_directory('rosbridge_server')
    rosbridge_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(rosbridge_server_pkg, 'launch', 'rosbridge_websocket_launch.xml')
        )
    )

    return LaunchDescription([
        web_interface_node,
        originman_action_node,
        image_publisher_node,
        image_caption_node,
        text_to_speech_node,
        rosbridge_server_node
    ])