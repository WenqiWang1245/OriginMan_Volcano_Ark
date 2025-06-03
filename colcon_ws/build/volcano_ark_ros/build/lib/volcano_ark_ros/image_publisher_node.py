import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImagePublisherNode(Node):
    def __init__(self):
        super().__init__('image_publisher_node')
        self.bridge = CvBridge()

        # 从ROS参数获取配置，提供默认值
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('publish_frequency', 5.0) # Hz

        camera_idx = self.get_parameter('camera_index').get_parameter_value().integer_value
        pub_freq = self.get_parameter('publish_frequency').get_parameter_value().double_value

        if pub_freq <= 0:
            self.get_logger().warn("发布频率必须大于0，将使用默认值 1.0 Hz")
            timer_period = 1.0
        else:
            timer_period = 1.0 / pub_freq # seconds

        self.publisher_ = self.create_publisher(Image, 'image_raw', 10)
        
        self.get_logger().info(f"尝试打开摄像头索引 {camera_idx}...")
        self.cap = cv2.VideoCapture(camera_idx)
        
        if not self.cap.isOpened():
            self.get_logger().error(f"无法打开摄像头索引 {camera_idx}。请检查摄像头是否连接或索引是否正确。")
            # 如果需要，可以在此尝试备用摄像头索引
            return # 如果摄像头打不开，节点无法工作

        self.get_logger().info(f"摄像头已打开 (索引 {camera_idx})。实际API后端: {self.cap.getBackendName()}")
        self.timer = self.create_timer(timer_period, self.publish_image)
        self.get_logger().info(f"图像发布节点已启动，将以 {pub_freq} Hz 的频率向 /image_raw 发布图像。")

    def publish_image(self):
        ret, frame = self.cap.read()
        if ret:
            try:
                ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                ros_image.header.stamp = self.get_clock().now().to_msg()
                ros_image.header.frame_id = "camera_optical_frame" 
                self.publisher_.publish(ros_image)
            except Exception as e:
                self.get_logger().error(f"转换或发布图像时出错: {str(e)}")
        else:
            self.get_logger().warn("无法从摄像头读取图像帧。")

    def destroy_node(self):
        self.get_logger().info("正在关闭摄像头...")
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    image_publisher_node = ImagePublisherNode()
    if not image_publisher_node.cap or not image_publisher_node.cap.isOpened():
        image_publisher_node.get_logger().fatal("摄像头初始化失败，节点将关闭。")
        image_publisher_node.destroy_node()
        rclpy.shutdown()
        return

    try:
        rclpy.spin(image_publisher_node)
    except KeyboardInterrupt:
        image_publisher_node.get_logger().info('用户中断，节点关闭。')
    finally:
        image_publisher_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()