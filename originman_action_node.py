# encoding:utf-8
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Empty
import time
import hiwonder.ActionGroupControl
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import os
import threading

ORIGINMAN_BASE_ACTIONS = [
    "stand", "go_forward", "back_fast", "left_move_fast", "right_move_fast",
    "push_ups", "sit_ups", "turn_left", "turn_right", "wave", "bow", "squat",
    "chest", "left_shot_fast", "right_shot_fast", "wing_chun", "left_uppercut",
    "right_uppercut", "left_kick", "right_kick", "stand_up_front",
    "stand_up_back", "twist", "stepping", "jugong", "weightlifting"
]
ORIGINMAN_DANCE_ACTIONS_NUMERIC_STR = [str(i) for i in range(16, 25)]

VALID_HW_ACTIONS = ORIGINMAN_BASE_ACTIONS + ORIGINMAN_DANCE_ACTIONS_NUMERIC_STR
ACTION_TYPE_SING = "sing_song"
AUDIO_BASE_PATH = "/userdata/OriginMan/audio/"
VALID_SONG_IDS = [str(i) for i in range(16, 25)]


class RobotActionController:
    def __init__(self, logger_object): 
        self._logger = logger_object
        self.current_song_process = None
        self.song_process_lock = threading.Lock() 

    def get_logger(self): 
        return self._logger

    def run_action_group(self, action_group_name, repetitions=1, wait_time_per_rep=None):
        self._logger.info(f"准备执行实体动作组: {action_group_name}, 重复次数: {repetitions}")
        if action_group_name not in VALID_HW_ACTIONS:
            self._logger.error(f"动作组 '{action_group_name}' 不是有效的hiwonder动作组名称。")
            return False
        total_success = True
        for i in range(repetitions):
            self._logger.info(f"执行实体动作 '{action_group_name}' (第 {i+1}/{repetitions} 次)")
            try:
                hiwonder.ActionGroupControl.runActionGroup(action_group_name) 
                estimated_time_single = 0.2 
                if wait_time_per_rep is not None:
                    estimated_time_single = wait_time_per_rep
                elif action_group_name == "push_ups": 
                    estimated_time_single = 0.5
                elif action_group_name in ORIGINMAN_DANCE_ACTIONS_NUMERIC_STR:
                    estimated_time_single = 0.5
                elif action_group_name in ["go_forward", "back_fast", "turn_left", "turn_right", "left_move_fast", "right_move_fast"]:
                    estimated_time_single = 0.2
                self._logger.info(f"实体动作 '{action_group_name}' (第 {i+1} 次) 执行中，等待 {estimated_time_single} 秒...")
                time.sleep(estimated_time_single)
                self._logger.info(f"实体动作 '{action_group_name}' (第 {i+1} 次) 执行完毕。")
            except Exception as e:
                self._logger.error(f"执行实体动作组 {action_group_name} (第 {i+1} 次) 时出错: {e}")
                total_success = False
                break 
        if repetitions > 1 or not total_success:
            self._logger.info(f"实体动作组 {action_group_name} (共{repetitions}次) 全部执行完毕。总体成功: {total_success}")
        return total_success


    def play_song(self, song_id):
        self._logger.info(f"准备播放歌曲，ID: {song_id}")
        if song_id not in VALID_SONG_IDS:
            self._logger.error(f"歌曲ID '{song_id}' 无效。可用ID: {VALID_SONG_IDS}")
            return False 

        song_file_name = f"{song_id}.wav"
        song_full_path = os.path.join(AUDIO_BASE_PATH, song_file_name)

        if not os.path.exists(song_full_path):
            self._logger.error(f"歌曲文件未找到: {song_full_path}")
            return False 

        aplay_command = ["aplay", "-q", song_full_path] 
        
        local_process_handle = None
        try:
            with self.song_process_lock:
                
                if self.current_song_process and self.current_song_process.poll() is None:
                    self._logger.warn("已有歌曲正在播放，新的播放请求被忽略。请先停止当前歌曲。") 
                    return False

                
                local_process_handle = subprocess.Popen(aplay_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.current_song_process = local_process_handle
                self._logger.info(f"歌曲 '{song_file_name}' 开始播放 (PID: {self.current_song_process.pid}).")

            
            return_code = local_process_handle.wait()
            
            with self.song_process_lock: 
                
                if self.current_song_process == local_process_handle:
                    self.current_song_process = None 
            
            if return_code == 0:
                self._logger.info(f"歌曲 '{song_file_name}' 正常播放完毕。")
                return True 
            elif return_code < 0 : 
                self._logger.info(f"歌曲 '{song_file_name}' 播放被终止 (信号: {return_code})。")
                return False 
            else: 
                
                self._logger.error(f"播放歌曲 '{song_file_name}' 时aplay返回错误码: {return_code}.")
                return False

        except FileNotFoundError:
            self._logger.error("未找到 aplay 命令。请确保已安装 alsa-utils 且 aplay 在 PATH 中。")
            with self.song_process_lock: # 确保清理
                if self.current_song_process == local_process_handle:
                    self.current_song_process = None
            return False
        except Exception as e:
            self._logger.error(f"播放歌曲 '{song_file_name}' 期间发生异常: {e}", exc_info=True)
            with self.song_process_lock: # 确保清理
                 if self.current_song_process == local_process_handle: 
                    self.current_song_process = None
            return False

    def cancel_current_song(self):
        with self.song_process_lock:
            if self.current_song_process and self.current_song_process.poll() is None: 
                self._logger.info(f"收到外部取消唱歌请求，正在终止aplay进程 (PID: {self.current_song_process.pid})...")
                self.current_song_process.terminate() 
                self._logger.info("终止信号已发送给aplay进程。")
                return True
            else:
                self._logger.info("没有正在播放的歌曲可以取消，或歌曲进程已结束。")
                return False

class OriginmanActionNode(Node):
    def __init__(self):
        super().__init__('originman_action_node')
        self.action_controller = RobotActionController(self.get_logger()) 
        self.action_executor = ThreadPoolExecutor(max_workers=1)

        self.action_subscription = self.create_subscription(
            String,
            '/robot_action_command',
            self.command_callback,
            10)
        
        self.cancel_singing_subscriber = self.create_subscription(
            Empty,
            '/robot_cancel_singing_request',
            self.cancel_singing_callback,
            10)
            
        self.get_logger().info(f'OriginMan 动作及唱歌节点已启动，等待指令 /robot_action_command 和 /robot_cancel_singing_request ...')
        self.get_logger().info(f'有效的HiWonder动作列表: {VALID_HW_ACTIONS}')
        self.get_logger().info(f'有效的歌曲ID列表: {VALID_SONG_IDS} (存放于: {AUDIO_BASE_PATH})')

        try:
            self.get_logger().info("节点初始化：执行 'stand' 动作让机器人站好。")
            self.action_controller.run_action_group("stand", repetitions=1, wait_time_per_rep=3) 
        except Exception as e:
            self.get_logger().error(f"初始化站立动作失败: {e}")

    def cancel_singing_callback(self, msg: Empty):
        self.get_logger().info("接收到停止唱歌的请求...")
        if self.action_controller.cancel_current_song():
            self.get_logger().info("尝试停止当前歌曲的操作已执行。")
        else:
            self.get_logger().info("当前没有歌曲在播放或无法停止。")


    def _execute_task_in_thread(self, action_type, action_details):
        action_name = action_details.get("action_name") 
        success = False 

        if action_type == "physical_action":
            repetitions = action_details.get("repetitions", 1)
            self.get_logger().info(f"线程池开始执行实体动作: '{action_name}', 重复: {repetitions} 次")
            success = self.action_controller.run_action_group(action_name, repetitions)
            self.get_logger().info(f"线程池完成实体动作: '{action_name}', 总体成功: {success}")
        
        elif action_type == ACTION_TYPE_SING:
            song_id = action_details.get("song_id")
            self.get_logger().info(f"线程池开始播放歌曲: ID '{song_id}' (动作名标记为 '{action_name}')")
            success = self.action_controller.play_song(song_id) 
            if success:
                self.get_logger().info(f"线程池完成歌曲播放: ID '{song_id}', 成功播放。")
            else:
                self.get_logger().warn(f"线程池歌曲播放: ID '{song_id}', 未能成功完成 (可能被取消或出错)。")
        else:
            self.get_logger().error(f"未知的任务类型 '{action_type}' 提交给线程池。")


    def command_callback(self, msg):
        
        raw_command_str = msg.data
        self.get_logger().info(f"接收到原始指令字符串: '{raw_command_str}'")
        try:
            payload = json.loads(raw_command_str)
            if not isinstance(payload, dict):
                self.get_logger().error(f"指令负载不是一个有效的JSON对象: {raw_command_str}")
                return
            action_name = payload.get("action_name", "").lower().strip()
            if not action_name:
                self.get_logger().error(f"指令JSON中缺少 'action_name': {payload}")
                return
            self.get_logger().info(f"解析后指令动作名: '{action_name}', 负载: {payload}")

            if action_name == ACTION_TYPE_SING:
                song_id = payload.get("song_id")
                if song_id and song_id in VALID_SONG_IDS:
                    task_details = {"action_name": action_name, "song_id": song_id}
                    self.action_executor.submit(self._execute_task_in_thread, ACTION_TYPE_SING, task_details)
                else:
                    self.get_logger().warn(f"唱歌指令 '{action_name}' 的 song_id '{song_id}' 无效或缺失。")
            elif action_name in VALID_HW_ACTIONS:
                repetitions = int(payload.get("repetitions", 1))
                if repetitions < 1: repetitions = 1
                task_details = {"action_name": action_name, "repetitions": repetitions}
                self.action_executor.submit(self._execute_task_in_thread, "physical_action", task_details)
            else:
                self.get_logger().warn(f"未知或不支持的动作名: '{action_name}' (来自指令: '{raw_command_str}')")
        except json.JSONDecodeError:
            self.get_logger().error(f"无法解析收到的指令为JSON: '{raw_command_str}'。")
        except Exception as e:
            self.get_logger().error(f"处理指令 '{raw_command_str}' 时发生未知错误: {e}", exc_info=True)

    def destroy_node(self):
        self.get_logger().info("正在关闭动作/唱歌节点线程池...")
        self.action_controller.cancel_current_song()
        self.action_executor.shutdown(wait=True) 
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    originman_action_node = OriginmanActionNode()
    try:
        rclpy.spin(originman_action_node)
    except KeyboardInterrupt:
        originman_action_node.get_logger().info('用户中断，节点关闭。')
    finally:
        originman_action_node.destroy_node()
        if rclpy.ok():
             rclpy.shutdown()

if __name__ == '__main__':
    main()