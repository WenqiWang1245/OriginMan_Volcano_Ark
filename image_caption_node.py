import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import base64
from volcenginesdkarkruntime import Ark
from std_msgs.msg import String
import threading
import requests

class ImageCaptionNode(Node):
    def __init__(self):
        super().__init__('image_caption_node')
        
        self.vision_model_id = os.environ.get("VOLCANO_VISION_MODEL", "doubao-1.5-vision-pro-250328")

        default_api_key_from_env = os.environ.get("ARK_API_KEY", "")
        self.declare_parameter('ark_api_key', default_api_key_from_env)
        
        self.api_key = self.get_parameter('ark_api_key').get_parameter_value().string_value
        
        self.add_on_set_parameters_callback(self._parameters_callback)

        self.ark_client = None
        if self.api_key:
            self.get_logger().info(f"ARK_API_KEY 参数已加载: ...{self.api_key[-4:] if len(self.api_key) > 4 else '****'}")
            self._initialize_ark_client()
        else:
            self.get_logger().warn("ARK_API_KEY 参数未设置或为空。节点可能无法调用视觉API，直到通过参数设置。")

        self.bridge = CvBridge()
        self.latest_image = None
        self.image_lock = threading.Lock()

        self.image_subscriber = self.create_subscription(
            Image,
            '/image_raw', 
            self.cache_image_callback,
            rclpy.qos.qos_profile_sensor_data
        )
        
        self.question_subscriber = self.create_subscription(
            String,
            '/vision_question',
            self.vision_question_callback,
            10
        )
        
        self.description_to_tts_publisher = self.create_publisher(String, '/tts_input', 10)
        self.description_to_web_publisher = self.create_publisher(String, '/web_display_text', 10)

        self.processing_lock = threading.Lock()

        self.get_logger().info('图像描述节点已启动。订阅 /image_raw 和 /vision_question。')
        self.get_logger().info(f'  视觉模型ID: {self.vision_model_id}')
        self.get_logger().info('  图像描述将发布到 /tts_input 和 /web_display_text。')

    def _initialize_ark_client(self):
        if self.api_key:
            try:
                self.ark_client = Ark(api_key=self.api_key)
                self.get_logger().info("Ark 客户端已成功初始化/更新。")
            except Exception as e:
                self.get_logger().error(f"初始化 Ark 客户端失败: {e}")
                self.ark_client = None
        else:
            self.get_logger().warn("API Key 未提供，Ark 客户端未初始化。")
            self.ark_client = None

    def _parameters_callback(self, params):
        changed = False
        for param in params:
            if param.name == 'ark_api_key':
                old_key_display = f"...{self.api_key[-4:] if len(self.api_key) > 4 else '****'}" if self.api_key else "未设置"
                self.api_key = param.value
                new_key_display = f"...{self.api_key[-4:] if len(self.api_key) > 4 else '****'}" if self.api_key else "未设置"
                self.get_logger().info(f"ARK_API_KEY 参数已从 '{old_key_display}' 更新为 '{new_key_display}'。")
                changed = True
        
        if changed:
            self._initialize_ark_client()
            
        return rclpy.parameter.SetParametersResult(successful=True)

    def cache_image_callback(self, msg):
        with self.image_lock:
            self.latest_image = msg

    def vision_question_callback(self, msg):
        user_prompt_text = msg.data
        self.get_logger().info(f"收到视觉问题: '{user_prompt_text}'")

        if not self.ark_client:
            self.get_logger().error("Ark 客户端未初始化 (API Key 可能未设置或无效)。无法处理视觉问题。")
            error_msg = String()
            error_msg.data = "抱歉，视觉服务配置不正确，我暂时无法回答视觉问题。"
            self.description_to_tts_publisher.publish(error_msg)
            self.description_to_web_publisher.publish(error_msg)
            return

        if not self.processing_lock.acquire(blocking=False):
            self.get_logger().warn("正在处理上一个视觉请求，请稍后再试。")
            return

        try:
            with self.image_lock:
                if self.latest_image is None:
                    self.get_logger().warn("没有可用的缓存图像来描述。")
                    no_image_msg = String()
                    no_image_msg.data = "抱歉，我现在看不到图像。"
                    self.description_to_tts_publisher.publish(no_image_msg)
                    self.description_to_web_publisher.publish(no_image_msg)
                    return
                try:
                    cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='bgr8')
                except CvBridgeError as e:
                    self.get_logger().error(f'CvBridge 转换错误: {e}')
                    return
            
            _, buffer = cv2.imencode('.jpg', cv_image)
            base64_image = base64.b64encode(buffer).decode('utf-8')
            final_prompt_for_llm = f"你是一个机器人助手。请根据提供的图片回答用户的问题。你的回答需要以“我看到了”或“我观察到”这四个字开始。用户的问题是：“{user_prompt_text}”"
            messages = [{"role": "user", "content": [{"type": "text", "text": final_prompt_for_llm},{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}]
            self.get_logger().info(f"向火山视觉模型 ({self.vision_model_id}) 发送请求...")
            completion = self.ark_client.chat.completions.create(model=self.vision_model_id, messages=messages)
            description = completion.choices[0].message.content.strip()
            
            if description:
                self.get_logger().info(f"生成的描述: {description}")
                desc_msg = String()
                desc_msg.data = description
                self.description_to_tts_publisher.publish(desc_msg)
                self.description_to_web_publisher.publish(desc_msg)
            else:
                self.get_logger().warn("火山视觉API未返回有效描述。")
                empty_desc_msg = String()
                empty_desc_msg.data = "抱歉，我无法描述所看到的画面。"
                self.description_to_tts_publisher.publish(empty_desc_msg)
                self.description_to_web_publisher.publish(empty_desc_msg)

        except requests.exceptions.Timeout:
            self.get_logger().error("调用火山视觉API超时。")
            timeout_msg = String()
            timeout_msg.data = "抱歉，视觉服务响应超时了。"
            self.description_to_tts_publisher.publish(timeout_msg)
            self.description_to_web_publisher.publish(timeout_msg)
        except Exception as e:
            self.get_logger().error(f'处理视觉问题或调用API时出错: {type(e).__name__} - {e}', exc_info=True)
            error_msg_str = String()
            error_msg_str.data = "抱歉，处理视觉请求时发生内部错误。"
            self.description_to_tts_publisher.publish(error_msg_str)
            self.description_to_web_publisher.publish(error_msg_str)
        finally:
            self.processing_lock.release()

def main(args=None):
    rclpy.init(args=args)
    image_caption_node = ImageCaptionNode()
    try:
        rclpy.spin(image_caption_node)
    except KeyboardInterrupt:
        image_caption_node.get_logger().info('用户中断，节点关闭。')
    finally:
        image_caption_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()