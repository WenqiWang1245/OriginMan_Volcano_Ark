#!/usr/bin/env python3
# encoding:utf-8
from flask import Flask, request, render_template_string, jsonify
import rclpy
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String, Empty
import threading
import json
import os
import requests
import sys
import random

# --- 配置 ---
TEXT_LLM_MODEL_NAME = os.environ.get("VOLCANO_TEXT_MODEL", "doubao-1.5-pro-32k-250115")
VOLCANO_CHAT_API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"

# --- 全局变量 ---
current_ark_api_key = None
app = Flask(__name__)

ros_node = None
ros_publisher_action = None
ros_publisher_vision = None
ros_publisher_tts_input = None
ros_publisher_cancel_singing = None
ros_speech_state_subscriber = None

last_speech_state = None
speech_state_lock = threading.Lock()
speech_tts_completion_event = threading.Event()

# --- OriginMan 机器人动作定义 ---
ORIGINMAN_BASE_ACTIONS = [
    "stand", "go_forward", "back_fast", "left_move_fast", "right_move_fast",
    "push_ups", "sit_ups", "turn_left", "turn_right", "wave", "bow", "squat",
    "chest", "left_shot_fast", "right_shot_fast", "wing_chun", "left_uppercut",
    "right_uppercut", "left_kick", "right_kick", "stand_up_front",
    "stand_up_back", "twist", "stepping", "jugong", "weightlifting"
]
ORIGINMAN_DANCE_ACTIONS_NUMERIC_STR = [str(i) for i in range(16, 25)]
ORIGINMAN_SONG_IDS_STR = [str(i) for i in range(16, 25)]

ORIGINMAN_ACTIONS = ORIGINMAN_BASE_ACTIONS + ORIGINMAN_DANCE_ACTIONS_NUMERIC_STR
ORIGINMAN_ACTIONS_CHINESE_MAP = {
    "stand": "立正", "go_forward": "前进", "back_fast": "后退",
    "left_move_fast": "左移", "right_move_fast": "右移", "push_ups": "俯卧撑",
    "sit_ups": "仰卧起坐", "turn_left": "左转", "turn_right": "右转",
    "wave": "挥手", "bow": "鞠躬", "squat": "下蹲", "chest": "庆祝",
    "left_shot_fast": "左脚踢", "right_shot_fast": "右脚踢", "wing_chun": "咏春",
    "left_uppercut": "左勾拳", "right_uppercut": "右勾拳", "left_kick": "左侧踢",
    "right_kick": "右侧踢", "stand_up_front": "前跌倒起立",
    "stand_up_back": "后跌倒起立", "twist": "扭腰", "stepping": "原地踏步",
    "jugong": "鞠躬", "weightlifting": "举重"
}
for dance_num_str in ORIGINMAN_DANCE_ACTIONS_NUMERIC_STR:
    ORIGINMAN_ACTIONS_CHINESE_MAP[dance_num_str] = f"舞蹈动作{dance_num_str}"

action_descriptions_for_prompt = "\n".join(
    [f"  - {en_action} (中文名: {ORIGINMAN_ACTIONS_CHINESE_MAP.get(en_action, en_action)})" for en_action in ORIGINMAN_ACTIONS]
)
song_ids_for_prompt = ", ".join(ORIGINMAN_SONG_IDS_STR)


# --- ROS 相关函数 ---
def speech_state_callback(msg):
    global last_speech_state, speech_tts_completion_event, ros_node
    with speech_state_lock:
        last_speech_state = msg.data
    if ros_node:
        ros_node.get_logger().debug(f"收到语音状态: {msg.data}")
    if msg.data == "正在聆听":
        speech_tts_completion_event.set()

def init_ros():
    global ros_node, ros_publisher_action, ros_publisher_vision, ros_publisher_tts_input
    global current_ark_api_key, ros_speech_state_subscriber, ros_publisher_cancel_singing
    ros_node = rclpy.create_node('web_interface_node')
    current_ark_api_key = os.environ.get("ARK_API_KEY")
    if current_ark_api_key:
        ros_node.get_logger().info("ARK_API_KEY 已从环境变量加载作为初始值。")
    else:
        ros_node.get_logger().warn("未在环境变量中找到 ARK_API_KEY。请通过Web界面进行设置。")
    ros_publisher_action = ros_node.create_publisher(String, '/robot_action_command', 10)
    ros_publisher_vision = ros_node.create_publisher(String, '/vision_question', 10)
    ros_publisher_tts_input = ros_node.create_publisher(String, '/tts_input', 10)
    ros_publisher_cancel_singing = ros_node.create_publisher(Empty, '/robot_cancel_singing_request', 10) 
    ros_speech_state_subscriber = ros_node.create_subscription(String, '/speech_state', speech_state_callback, 10)
    ros_node.get_logger().info('Web接口ROS节点相关组件已初始化。')

def call_intent_parsing_llm(user_query):
    global ros_node, current_ark_api_key
    log_source = ros_node.get_logger() if ros_node else print
    log_source.info(f"向火山文本LLM发送查询: '{user_query}'")
    if not current_ark_api_key:
        log_source.error("ARK_API_KEY 未配置！")
        return [{"type": "error", "message": "服务端错误：API Key未配置"}]
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {current_ark_api_key}"}
    system_prompt = f"""你是一个基于火山大模型的机器人助手，帮助用户控制名为OriginMan的机器人。
分析用户的指令，并将其转换为结构化的JSON命令。
机器人有以下主要能力：
1. "query_vision": 提出关于机器人所见内容的问题。结果应包含 "prompt_for_vision" 字段，内容为用户原始的视觉问题，识别的结果可以稍微简洁。
2. "perform_action": 执行一个物理动作。结果应包含 "action_name" 字段。
   对于像 "前进", "后退" 这样的动作，如果用户指定了次数（例如“前进两步”，“后退3次”），你应该在JSON中包含一个 "repetitions" 字段，其值为数字。如果未指定次数，则 "repetitions" 默认为 1。
   机器人能够执行以下物理动作。在生成JSON时，"action_name"字段必须严格使用括号前的英文指令名 (例如 "go_forward", 或对于舞蹈动作使用如 "16", "17" 这样的数字字符串):
{action_descriptions_for_prompt}
   当你需要生成 "perform_action" 类型的JSON时，"action_name" 字段必须严格使用上述列表中的英文指令名。 "repetitions" 字段应为整数。
3. "sing_song": 让机器人唱歌。结果应包含 "song_id" 字段，其值为歌曲的编号字符串 (例如 "16", "17", ..., "24")。
   可用的歌曲编号有: {song_ids_for_prompt}。
   如果用户说“唱首歌”或“来一首歌”等没有指定具体歌曲的指令，你必须从可用的歌曲编号中随机选择一个。不要询问用户。
4. "cancel_singing": 当用户说“别唱了”、“停止唱歌”、“安静”等意图停止当前正在播放的歌曲时，使用此类型。此类型不需要额外参数。
5. "direct_response": 如果用户的指令是问候、简单聊天、询问机器人能力（例如“你能做什么动作？”）或无法归类为上述类别，你可以直接生成一个回答。结果应包含 "text_to_speak" 字段。

当你回答用户关于可执行动作的问题时 (例如用户问：“你会做什么？”或“有哪些动作？”)，你应该在 "text_to_speak" 字段中用中文列出这些动作的中文名，并选择性的列出来，不需要全部列出。例如：“我会的动作有立正、前进、后退、挥手、鞠躬、各种舞蹈动作、我还会唱歌呢，等等。”。你不需要明确向用户说明“舞蹈动作16”、“歌曲17”这些具体的编号，只需要和用户说你会不同的舞蹈动作和歌曲就好。
如果用户发出一个模糊的跳舞请求，比如“跳个舞”或“表演一个舞蹈”，甚至说“做一些炫酷的动作”等，而没有指明具体是哪一个，你必须从机器人可执行的舞蹈动作编号（即 "16" 到 "24"）中自行选择一个，并生成 "perform_action" 类型的JSON。例如，你可以选择 "17"或者"20"等等。不要询问用户。确保 action_name 是一个有效的数字字符串。

请总是输出一个JSON对象列表，每个对象代表一个步骤。
如果用户指令包含多个步骤（例如“你看到了什么然后鞠个躬再唱首歌然后停掉”），请为每个步骤生成一个JSON对象。
示例:
用户: "桌子上有什么？然后鞠个躬，再唱一首歌" ->
[
  {{"type": "query_vision", "prompt_for_vision": "桌子上有什么？"}},
  {{"type": "perform_action", "action_name": "bow", "repetitions": 1}},
  {{"type": "sing_song", "song_id": "18"}}
]
用户: "前进两步" -> [{{"type": "perform_action", "action_name": "go_forward", "repetitions": 2}}]
用户: "唱首歌" -> [{{"type": "sing_song", "song_id": "{random.choice(ORIGINMAN_SONG_IDS_STR)}"}}]
用户: "别唱了" -> [{{"type": "cancel_singing"}}]
用户: "你好" -> [{{"type": "direct_response", "text_to_speak": "你好！有什么可以帮您的吗？"}}]
用户: "你能做什么动作" -> [{{"type": "direct_response", "text_to_speak": "我可以做的动作包括：立正、前进、后退、挥手、鞠躬、各种舞蹈以及唱歌等等。"}}]

如果无法理解或指令无效，请返回 [{{"type": "error", "message": "无法理解的指令"}}].
确保你的输出严格遵守JSON格式，并且是一个列表。"""
    final_system_prompt = system_prompt.replace("{random.choice(ORIGINMAN_DANCE_ACTIONS_NUMERIC_STR)}", random.choice(ORIGINMAN_DANCE_ACTIONS_NUMERIC_STR))
    final_system_prompt = final_system_prompt.replace("{random.choice(ORIGINMAN_SONG_IDS_STR)}", random.choice(ORIGINMAN_SONG_IDS_STR))

    data = {"model": TEXT_LLM_MODEL_NAME, "messages": [{"role": "system", "content": final_system_prompt}, {"role": "user", "content": user_query}]}
    try:
        response = requests.post(VOLCANO_CHAT_API_URL, headers=headers, json=data, timeout=20)
        response.raise_for_status()
        llm_response_json = response.json()
        if llm_response_json.get("choices") and len(llm_response_json["choices"]) > 0:
            content_text = llm_response_json["choices"][0].get("message", {}).get("content", "")
            log_source.info(f"火山LLM原始回复内容: {content_text}")
            if content_text.startswith("```json"): content_text = content_text[len("```json"):].strip()
            if content_text.endswith("```"): content_text = content_text[:-len("```")].strip()
            try:
                parsed_commands = json.loads(content_text)
                if not isinstance(parsed_commands, list):
                    log_source.warn(f"LLM未返回列表，而是: {type(parsed_commands)}. 将其包装在列表中。")
                    return [parsed_commands]
                return parsed_commands
            except json.JSONDecodeError as e:
                log_source.error(f"LLM输出的JSON解析失败: {e}. 内容: {content_text}")
                return [{"type": "error", "message": f"LLM输出格式错误: {content_text}"}]
        else:
            log_source.error(f"火山LLM API响应格式不符合预期: {llm_response_json}")
            return [{"type": "error", "message": "LLM API响应格式错误"}]
    except requests.exceptions.Timeout:
        log_source.error("调用火山LLM API超时。")
        return [{"type": "error", "message": "LLM API请求超时"}]
    except requests.exceptions.RequestException as e:
        log_source.error(f"调用火山LLM API时发生网络或HTTP错误: {e}")
        return [{"type": "error", "message": f"LLM API请求错误: {str(e)}"}]
    except Exception as e:
        log_source.error(f"调用火山LLM时发生未知异常: {e}")
        return [{"type": "error", "message": f"调用LLM异常: {str(e)}"}]

# --- HTML模板 ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>OriginMan 火山引擎AI助手</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://static.robotwebtools.org/roslibjs/current/roslib.min.js"></script>
    <style>
        :root { 
            --bg-main: #2F3640; --bg-container: #2a2d35; --bg-chatlog: #252830; --bg-input-area: #313540;
            --text-primary: #d1d5db; --text-secondary: #9ca3af; --text-title: #e5e7eb;
            --accent-color: #3b82f6; --accent-hover: #2563eb; --bot-message-bg: #374151;
            --error-message-bg: #4b1f24; --error-text-color: #fca5a5; --border-color: #4b5563;
            --border-focus-color: var(--accent-color);
            --font-family: 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            --border-radius-sm: 4px; --border-radius-md: 8px;
        }
        body {font-family: var(--font-family); margin: 0; padding: 20px; background-color: var(--bg-main); color: var(--text-primary); display: flex; justify-content: center; align-items: flex-start; min-height: 100vh; box-sizing: border-box;}
        .container {background-color: var(--bg-container); padding: 20px 25px; border-radius: var(--border-radius-md); border: 1px solid var(--border-color); width: 100%; max-width: 720px; display: flex; flex-direction: column; height: calc(100vh - 40px); max-height: 900px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);}
        h1 {text-align: center; color: var(--text-title); margin-top: 5px; margin-bottom: 15px; font-weight: 300; font-size: 1.75em; letter-spacing: 0.5px; border-bottom: 1px solid var(--border-color); padding-bottom: 15px;}
        .config-area {margin-bottom: 15px; padding: 15px; background-color: var(--bg-input-area); border-radius: var(--border-radius-sm); border: 1px solid var(--border-color);}
        .config-area label {display: block; margin-bottom: 5px; color: var(--text-secondary); font-size: 0.9em;}
        .config-area input[type="text"] {width: calc(100% - 24px); padding: 8px 10px; margin-bottom: 10px; border-radius: var(--border-radius-sm); border: 1px solid var(--border-color); background-color: var(--bg-chatlog); color: var(--text-primary); font-size: 0.9em;}
        .config-area button {background-color: var(--accent-color); color: white; padding: 8px 15px; border: none; border-radius: var(--border-radius-sm); cursor: pointer; font-size: 0.9em; transition: background-color 0.2s;}
        .config-area button:hover {background-color: var(--accent-hover);}
        #chatLog {flex-grow: 1; margin-bottom: 15px; padding: 15px; background-color: var(--bg-chatlog); border-radius: var(--border-radius-sm); overflow-y: auto; border: 1px solid var(--border-color); display: flex; flex-direction: column; gap: 10px; scrollbar-width: thin; scrollbar-color: var(--border-color) var(--bg-chatlog);}
        #chatLog::-webkit-scrollbar {width: 8px;} #chatLog::-webkit-scrollbar-track {background: var(--bg-chatlog); border-radius: var(--border-radius-sm);}
        #chatLog::-webkit-scrollbar-thumb {background-color: var(--border-color); border-radius: var(--border-radius-sm); border: 2px solid var(--bg-chatlog);}
        #chatLog::-webkit-scrollbar-thumb:hover {background-color: var(--text-secondary);}
        .message {padding: 10px 14px; border-radius: var(--border-radius-md); max-width: 80%; word-wrap: break-word; line-height: 1.6; font-size: 0.9em; opacity: 0; transform: translateY(10px); animation: messageFadeIn 0.3s ease-out forwards;}
        @keyframes messageFadeIn {to {opacity: 1; transform: translateY(0);}}
        .user-message {background-color: var(--accent-color); color: white; align-self: flex-end; border-bottom-right-radius: var(--border-radius-sm);}
        .bot-message {background-color: var(--bot-message-bg); color: var(--text-primary); align-self: flex-start; border-bottom-left-radius: var(--border-radius-sm);}
        .error-message {background-color: var(--error-message-bg); color: var(--error-text-color); align-self: flex-start; border-bottom-left-radius: var(--border-radius-sm);}
        .input-area {display: flex; gap: 8px; padding: 10px; background-color: var(--bg-input-area); border-radius: var(--border-radius-md); margin-top: auto; border: 1px solid var(--border-color);}
        textarea#naturalLanguageInput {flex-grow: 1; padding: 10px 12px; border-radius: var(--border-radius-sm); border: 1px solid var(--border-color); background-color: var(--bg-chatlog); color: var(--text-primary); font-size: 0.95em; min-height: 44px; resize: none; outline: none; transition: border-color 0.2s ease-in-out, box-shadow 0.2s ease-in-out;}
        textarea#naturalLanguageInput:focus {border-color: var(--border-focus-color); box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3);}
        textarea#naturalLanguageInput::placeholder {color: var(--text-secondary); font-style: italic;}
        button#sendButton {background-color: var(--accent-color); color: white; padding: 0 20px; border: none; border-radius: var(--border-radius-sm); cursor: pointer; font-size: 0.95em; font-weight: 500; transition: background-color 0.2s ease-in-out; min-height: 44px; display: flex; align-items: center; justify-content: center;}
        button#sendButton:hover {background-color: var(--accent-hover);}
        button#sendButton:active {transform: translateY(1px);}
    </style>
</head>
<body>
    <div class="container">
        <h1>OriginMan 火山引擎AI助手</h1>
        <div class="config-area">
            <label for="arkApiKeyInput">火山 ARK API Key:</label>
            <input type="text" id="arkApiKeyInput" placeholder="在此输入或更新 ARK API Key">
            <button onclick="saveApiKey()">保存 API Key</button>
        </div>
        <div id="chatLog">
            <div class="message bot-message">你好！我是OriginMan火山引擎助手，已准备就绪。我支持的动作有立正，前进，后退，左移，右移，俯卧撑，仰卧起坐，左转，右转，挥手，鞠躬，下蹲，庆祝，左脚踢，右脚踢，咏春，左勾拳，右勾拳，左侧踢，右侧踢，各种舞蹈，我还会唱歌呢！欢迎和我进行互动！</div>
        </div>
        <div class="input-area">
            <textarea id="naturalLanguageInput" placeholder="在此输入指令 (如: '跳个舞', '向前走3步', '唱首歌', '别唱了', '你看到了什么'等)..." rows="1"></textarea>
            <button id="sendButton" onclick="sendNaturalLanguageCommand()">发送</button>
        </div>
    </div>

    <script>
        function appendMessage(text, type) {
            const chatLog = document.getElementById('chatLog');
            const messageDiv = document.createElement('div');
            messageDiv.classList.add('message', type); 
            messageDiv.textContent = text; 
            chatLog.appendChild(messageDiv);
            setTimeout(() => { 
                chatLog.scrollTop = chatLog.scrollHeight; 
            }, 50); 
        }

        async function saveApiKey() {
            const apiKey = document.getElementById('arkApiKeyInput').value;
            if (!apiKey.trim()) {
                alert('请输入有效的 ARK API Key。');
                return;
            }
            try {
                const response = await fetch('/save_api_key', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ark_api_key: apiKey })
                });
                const result = await response.json();
                if (result.status === 'success') {
                    alert(result.message || 'API Key 已成功保存！');
                } else {
                    alert('API Key 保存失败: ' + (result.message || '未知错误'));
                }
            } catch (error) {
                alert('保存 API Key 时发生网络错误: ' + error);
            }
        }

        async function sendNaturalLanguageCommand() {
            const userInputElement = document.getElementById('naturalLanguageInput');
            const userInput = userInputElement.value;
            if (!userInput.trim()) { return; }

            appendMessage(userInput, 'user-message');
            userInputElement.value = ''; 
            userInputElement.focus(); 

            try {
                const response = await fetch('/execute_natural_language_command', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ command: userInput })
                });
                const result = await response.json();
                let botResponseText = "";
                let messageType = 'bot-message';

                if (result.error) {
                    botResponseText = `错误: ${result.error}`;
                    messageType = 'error-message';
                } else {
                    if (result.llm_direct_response_text) {
                        botResponseText = result.llm_direct_response_text;
                    } else if (result.executed_actions && result.executed_actions.length > 0) {
                        let firstActionDisplayText = result.executed_actions[0].details && result.executed_actions[0].details.display_text;
                        if (firstActionDisplayText) {
                             botResponseText = firstActionDisplayText;
                        } else {
                             botResponseText = (result.status === "已接收处理") ? "指令已接收，正在后台处理..." : "指令状态未知或无文本回复。";
                        }
                        if (result.executed_actions.some(act => act.type === "llm_error" || act.type === "unknown_command_type")){
                            messageType = 'error-message';
                            const errorAction = result.executed_actions.find(act => act.type === "llm_error" || act.type === "unknown_command_type");
                            if(errorAction && errorAction.details && errorAction.details.message) {
                                botResponseText = `错误: ${errorAction.details.message}`;
                            } else if (errorAction && errorAction.details && errorAction.details.display_text){
                                botResponseText = `错误: ${errorAction.details.display_text}`;
                            } else {
                                botResponseText = "处理指令时发生未知错误。";
                            }
                        }
                    } else { 
                        botResponseText = (result.status === "已接收处理") ? "指令已接收，正在后台处理..." : "未收到明确回复或动作。";
                    }
                }
                
                if (botResponseText && botResponseText.trim()) {
                    appendMessage(botResponseText, messageType);
                } else if (messageType !== 'error-message' && result.status === "已接收处理") { 
                    appendMessage("指令已发送，后台处理中。", "bot-message");
                }
            } catch (error) {
                appendMessage('与机器人通信失败: ' + error.message, 'error-message');
            }
        }
        
        document.getElementById('naturalLanguageInput').addEventListener('keypress', function (e) {
            if (e.key === 'Enter' && !e.shiftKey) {e.preventDefault(); sendNaturalLanguageCommand();}
        });
        // ORIGINMAN_ACTIONS_CHINESE_MAP 这个JavaScript变量之前用于前端显示，现在回复语主要由后端生成，前端不再直接使用此映射。
        
    </script>
</body>
</html>
"""

# --- Flask 路由 ---
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/save_api_key', methods=['POST'])
def save_api_key_route():
    global current_ark_api_key, ros_node
    data = request.get_json()
    new_api_key = data.get('ark_api_key')
    if not new_api_key or not new_api_key.strip():
        return jsonify({"status": "error", "message": "API Key 不能为空"}), 400
    current_ark_api_key = new_api_key.strip()
    log_msg = f"ARK_API_KEY 已在Web服务中更新为: ...{current_ark_api_key[-4:] if len(current_ark_api_key) > 4 else '****'}"
    if ros_node: ros_node.get_logger().info(log_msg)
    else: print(log_msg)
    return jsonify({"status": "success", "message": "API Key 已在本服务中保存。"})

# --- 后台执行函数 ---
def _handle_sequential_execution_in_background(commands_for_execution):
    global ros_node, ros_publisher_tts_input, ros_publisher_action, ros_publisher_vision, speech_tts_completion_event
    if not ros_node:
        print("[后台线程错误] ros_node 未初始化！")
        return
    ros_node.get_logger().info(f"[后台线程] 开始执行任务序列: {commands_for_execution}")
    for cmd_detail in commands_for_execution:
        cmd_type = cmd_detail.get("type")
        data = cmd_detail.get("data")
        if cmd_type == "direct_response_for_tts":
            text_to_speak = data.get("text_to_speak")
            if text_to_speak and ros_publisher_tts_input:
                speech_tts_completion_event.clear()
                tts_msg = String()
                tts_msg.data = text_to_speak
                ros_publisher_tts_input.publish(tts_msg)
                ros_node.get_logger().info(f"[后台线程] TTS指令已发送: '{text_to_speak}', 等待播报完成...")
                if speech_tts_completion_event.wait(timeout=60.0): 
                    ros_node.get_logger().info("[后台线程] TTS播报完成。")
                else:
                    ros_node.get_logger().warn("[后台线程] TTS播报等待超时。")
        elif cmd_type == "perform_action_later":
            action_name = data.get("action_name")
            repetitions = data.get("repetitions", 1)
            if action_name and action_name in ORIGINMAN_ACTIONS and ros_publisher_action:
                action_payload = {"action_name": action_name, "repetitions": repetitions}
                msg_str = json.dumps(action_payload)
                msg = String()
                msg.data = msg_str
                ros_publisher_action.publish(msg)
                ros_node.get_logger().info(f"[后台线程] 动作指令已发送: {msg_str}")
            else:
                ros_node.get_logger().warn(f"[后台线程] 无效动作 '{action_name}' 或发布器未就绪，跳过执行。")
        elif cmd_type == "sing_song_later":
            song_id = data.get("song_id")
            if song_id and ros_publisher_action:
                action_payload = {"action_name": "sing_song", "song_id": song_id}
                msg_str = json.dumps(action_payload)
                msg = String()
                msg.data = msg_str
                ros_publisher_action.publish(msg)
                ros_node.get_logger().info(f"[后台线程] 唱歌指令已发送: {msg_str}")
            else:
                ros_node.get_logger().warn(f"[后台线程] 无效歌曲ID '{song_id}' 或发布器未就绪，跳过演唱。")
        elif cmd_type == "query_vision_later":
            prompt = data.get("prompt_for_vision")
            if prompt and ros_publisher_vision:
                msg = String()
                msg.data = prompt
                ros_publisher_vision.publish(msg)
                ros_node.get_logger().info(f"[后台线程] 视觉问题已发送: {prompt}")
            else:
                ros_node.get_logger().warn(f"[后台线程] 视觉问题为空或发布器未就绪，跳过。")
        
    ros_node.get_logger().info("[后台线程] 所有后台任务执行完毕。")

@app.route('/execute_natural_language_command', methods=['POST'])
def execute_natural_language_command():
    global ros_node, current_ark_api_key, ros_publisher_cancel_singing
    data = request.get_json()
    user_command_text = data.get('command')
    if not user_command_text: return jsonify({"status": "失败", "error": "未提供指令文本"}), 400
    if not current_ark_api_key: return jsonify({"status": "失败", "error": "ARK API Key 未配置。"}), 400
    if not ros_node: return jsonify({"status": "失败", "error": "ROS节点服务未就绪。"}), 500

    parsed_commands_from_llm = call_intent_parsing_llm(user_command_text)
    executed_actions_info_for_web = [] 
    sequential_execution_plan_for_bg = [] 
    primary_text_for_web_display = None
    is_llm_error_only = False

    if not parsed_commands_from_llm:
        return jsonify({"status": "失败", "error": "LLM未能解析指令或调用失败", "executed_actions": []}), 500

    if len(parsed_commands_from_llm) == 1 and parsed_commands_from_llm[0].get("type") == "error":
        is_llm_error_only = True
        error_message = parsed_commands_from_llm[0].get("message", "LLM返回未知错误")
        
        primary_text_for_web_display = f"抱歉，处理您的指令时遇到问题: {error_message}"
    else:
        for cmd_idx, cmd in enumerate(parsed_commands_from_llm):
            cmd_type = cmd.get("type")
            friendly_display_text_for_action = "好的，马上为您处理！"

            if cmd_type == "direct_response":
                text_to_speak = cmd.get("text_to_speak")
                if text_to_speak:
                    if primary_text_for_web_display is None: primary_text_for_web_display = text_to_speak
                    
                    sequential_execution_plan_for_bg.append({"type": "direct_response_for_tts", "data": {"text_to_speak": text_to_speak}})
            
            elif cmd_type == "perform_action":
                action_name = cmd.get("action_name")
                repetitions = cmd.get("repetitions", 1)
                if not isinstance(repetitions, int) or repetitions < 1: repetitions = 1
                action_display_name = ORIGINMAN_ACTIONS_CHINESE_MAP.get(action_name, action_name)
                if action_name in ORIGINMAN_DANCE_ACTIONS_NUMERIC_STR:
                    friendly_display_text_for_action = f"没问题，准备为您表演舞蹈！"
                else:
                    friendly_display_text_for_action = f"正在执行..."
                if primary_text_for_web_display is None: primary_text_for_web_display = friendly_display_text_for_action
                executed_actions_info_for_web.append({"type": "perform_action", "details": {"action": action_name, "repetitions": repetitions, "display_text": friendly_display_text_for_action, "status": "待执行"}})
                sequential_execution_plan_for_bg.append({"type": "perform_action_later", "data": {"action_name": action_name, "repetitions": repetitions}})

            elif cmd_type == "sing_song":
                song_id = cmd.get("song_id")
                if song_id in ORIGINMAN_SONG_IDS_STR:
                    friendly_display_text_for_action = f"太棒了！准备为您演唱歌曲！"
                    if primary_text_for_web_display is None: primary_text_for_web_display = friendly_display_text_for_action
                    executed_actions_info_for_web.append({"type": "sing_song", "details": {"song_id": song_id, "display_text": friendly_display_text_for_action, "status": "待演唱"}})
                    sequential_execution_plan_for_bg.append({"type": "sing_song_later", "data": {"song_id": song_id}})
                else:
                    error_message = f"歌曲无效或未找到。"
                    if primary_text_for_web_display is None: primary_text_for_web_display = f"抱歉，{error_message}"
                    
                    ros_node.get_logger().warn(f"LLM请求演唱无效歌曲ID: {song_id}")
                    
                    is_llm_error_only = True 
                    break 
            
            elif cmd_type == "cancel_singing": 
                if ros_publisher_cancel_singing:
                    ros_publisher_cancel_singing.publish(Empty())
                    primary_text_for_web_display = "好的，已发送停止唱歌的请求！"
                    ros_node.get_logger().info("取消唱歌指令已发送。")
                    
                else:
                    primary_text_for_web_display = "抱歉，现在无法发送停止唱歌的请求。"
                
                sequential_execution_plan_for_bg = [] 
                break


            elif cmd_type == "query_vision":
                prompt = cmd.get("prompt_for_vision")
                friendly_display_text_for_action = f"好的，我来看看！"
                if primary_text_for_web_display is None: primary_text_for_web_display = friendly_display_text_for_action
                executed_actions_info_for_web.append({"type": "query_vision", "details": {"prompt": prompt, "display_text": friendly_display_text_for_action, "status": "待发送"}})
                sequential_execution_plan_for_bg.append({"type": "query_vision_later", "data": {"prompt_for_vision": prompt}})
            
            elif cmd_type == "error": 
                error_message = cmd.get("message", "LLM处理部分指令时出错")
                if primary_text_for_web_display is None: primary_text_for_web_display = f"处理时遇到点小麻烦: {error_message}"
                
                ros_node.get_logger().error(f"LLM指令中包含错误: {error_message}")
                is_llm_error_only = True
                sequential_execution_plan_for_bg = [] 
                break 
            
            else:
                unknown_cmd_text = f"我不确定怎么处理“{cmd_type}”这个指令呢。"
                if primary_text_for_web_display is None: primary_text_for_web_display = unknown_cmd_text
                
                ros_node.get_logger().warn(f"LLM解析: 未知指令类型 '{cmd_type}'。 CMD: {cmd}")
                is_llm_error_only = True 
                sequential_execution_plan_for_bg = []
                break
    
    if primary_text_for_web_display is None and sequential_execution_plan_for_bg:
        primary_text_for_web_display = "好的，指令已收到，我这就去处理！"
    elif primary_text_for_web_display is None and not sequential_execution_plan_for_bg and not is_llm_error_only:
         primary_text_for_web_display = "嗯...我好像没完全明白您的意思，可以换个说法吗？"

    if sequential_execution_plan_for_bg and not is_llm_error_only: 
        ros_node.get_logger().info("启动后台线程处理TTS、动作和唱歌...")
        bg_thread = threading.Thread(target=_handle_sequential_execution_in_background, args=(list(sequential_execution_plan_for_bg),))
        bg_thread.daemon = True
        bg_thread.start()
    elif is_llm_error_only: ros_node.get_logger().info("LLM直接返回错误或指令包含错误，不启动后台执行线程。")
    elif not sequential_execution_plan_for_bg: ros_node.get_logger().info("没有需要后台执行的指令(可能因错误或取消指令清空)。")
    else: ros_node.get_logger().info("没有需要后台执行的指令。")


    response_payload = {
        "executed_actions": executed_actions_info_for_web, 
        "llm_direct_response_text": primary_text_for_web_display
    }
    
    if is_llm_error_only or (primary_text_for_web_display and "错误" in primary_text_for_web_display) or \
       (primary_text_for_web_display and "抱歉" in primary_text_for_web_display and not sequential_execution_plan_for_bg and not executed_actions_info_for_web):
        response_payload["status"] = "失败" 
        response_payload["error"] = primary_text_for_web_display 
    else:
        response_payload["status"] = "已接收处理"
    
    return jsonify(response_payload), 200

# --- 主函数和ROS Spin线程 ---
def ros_spin_thread_func(node_to_spin):
    executor = MultiThreadedExecutor()
    executor.add_node(node_to_spin)
    logger = node_to_spin.get_logger() if hasattr(node_to_spin, 'get_logger') else print 
    logger.info("ROS Spin线程已启动。")
    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        logger.info("ROS Spin线程收到关闭信号。")
    except Exception as e:
        if hasattr(node_to_spin, '_Node__destroyed') and not node_to_spin._Node__destroyed and rclpy.ok():
            logger.error(f"ROS Spin线程发生异常: {e}", exc_info=True)
    finally:
        logger.info("ROS Spin线程已结束。")

def main(args=None): 
    global ros_node 
    rclpy.init(args=args) 
    init_ros() 
    if not ros_node:
        print("严重错误: ROS节点未能初始化。程序退出。", file=sys.stderr)
        if rclpy.ok(): rclpy.shutdown()
        return

    ros_thread = threading.Thread(target=ros_spin_thread_func, args=(ros_node,), daemon=True)
    ros_thread.start()
    ros_node.get_logger().info("启动Flask Web服务器，网页地址为机器人背后ip地址:5000")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except SystemExit:
        ros_node.get_logger().info("Flask服务器被终止 (SystemExit)。")
    except Exception as e:
        ros_node.get_logger().fatal(f"Web服务器启动或运行期间发生严重错误: {e}", exc_info=True)
    finally:
        if ros_node and rclpy.ok() and hasattr(ros_node, '_Node__destroyed') and not ros_node._Node__destroyed:
            ros_node.get_logger().info("请求关闭Web接口ROS节点...")
            ros_node.destroy_node() 
        
        if 'ros_thread' in locals() and ros_thread.is_alive():
            log_source = None
            if ros_node and hasattr(ros_node, 'get_logger') and hasattr(ros_node, '_Node__destroyed') and not ros_node._Node__destroyed:
                log_source = ros_node.get_logger()
            else:
                log_source = print 

            log_source.info("等待ROS spin线程结束...")
            ros_thread.join(timeout=2.0) 
            if ros_thread.is_alive(): log_source.warn("ROS spin线程未在超时内结束。")
        
        if rclpy.ok(): 
            rclpy.shutdown()
        print("Web接口服务已关闭。")

if __name__ == '__main__':
    main()