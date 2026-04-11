"""
Qwen Agent 核心模块
基于 DashScope OpenAI 兼容接口实现 LLM Agent 的工具调用循环
"""

import json
import os
from openai import OpenAI

from robot.base import RobotInterface
from agent.tools import TOOL_DEFINITIONS, execute_tool
from agent.prompts import SYSTEM_PROMPT, TASK_PROMPT_TEMPLATE


class GraspingAgent:
    """基于 Qwen 的自主抓取 Agent

    通过 Function Calling 机制，Agent 可以自主规划并调用工具
    控制机械臂完成抓取和放置任务。
    """

    # 最大工具调用轮次（防止无限循环）
    MAX_TOOL_ROUNDS = 20

    def __init__(
        self,
        robot: RobotInterface,
        model: str = "qwen-max",
        api_key: str = None,
        base_url: str = None,
        verbose: bool = True,
    ):
        """初始化 Agent

        Args:
            robot: 机器人接口实例
            model: Qwen 模型名称（qwen-max, qwen-plus, qwen-turbo）
            api_key: DashScope API Key（默认从环境变量读取）
            base_url: API Base URL
            verbose: 是否打印详细日志
        """
        self.robot = robot
        self.model = model
        self.verbose = verbose

        # 初始化 OpenAI 客户端（指向 DashScope）
        if api_key is None:
            api_key = os.getenv("DASHSCOPE_API_KEY")
        if api_key is None:
            raise ValueError(
                "未设置 DASHSCOPE_API_KEY 环境变量。\n"
                "请通过以下方式设置:\n"
                "  export DASHSCOPE_API_KEY='your-api-key'\n"
                "或在初始化时传入 api_key 参数。\n"
                "API Key 获取地址: https://bailian.console.aliyun.com/"
            )

        if base_url is None:
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        # 初始化对话历史
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        print(f"[Agent] Qwen Agent 初始化完成 (模型: {model})")

    def run(self, user_instruction: str) -> str:
        """执行用户指令

        主循环：
        1. 将用户指令发送给 Qwen
        2. Qwen 返回文本回复或工具调用
        3. 如果是工具调用：执行工具 → 将结果发送回 Qwen → 重复
        4. 如果是文本回复：返回最终结果

        Args:
            user_instruction: 用户的任务指令

        Returns:
            Agent 的最终回复文本
        """
        # 构造用户消息
        task_message = TASK_PROMPT_TEMPLATE.format(task=user_instruction)
        self.messages.append({"role": "user", "content": task_message})

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[Agent] 用户指令: {user_instruction}")
            print(f"{'='*60}")

        # 开始 Agent 循环
        for round_num in range(self.MAX_TOOL_ROUNDS):
            if self.verbose:
                print(f"\n--- Agent 轮次 {round_num + 1} ---")

            # 调用 Qwen API
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=TOOL_DEFINITIONS,
                    temperature=0.1,  # 低温度保证稳定性
                )
            except Exception as e:
                error_msg = f"API 调用失败: {str(e)}"
                print(f"[Agent] {error_msg}")
                return error_msg

            choice = response.choices[0]
            message = choice.message

            # 检查是否有工具调用
            if message.tool_calls:
                # 将 assistant 消息加入历史
                self.messages.append(message)

                # 执行所有工具调用
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                    if self.verbose:
                        print(f"[Agent] 调用工具: {tool_name}({json.dumps(arguments, ensure_ascii=False)})")

                    # 执行工具
                    result = execute_tool(tool_name, arguments, self.robot)

                    if self.verbose:
                        # 截断过长的结果显示
                        display_result = result[:500] + "..." if len(result) > 500 else result
                        print(f"[Agent] 工具结果: {display_result}")

                    # 将工具结果加入消息历史
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })

            elif message.content:
                # Agent 返回了最终文本回复
                final_response = message.content
                self.messages.append({
                    "role": "assistant",
                    "content": final_response,
                })

                if self.verbose:
                    print(f"\n{'='*60}")
                    print(f"[Agent] 最终回复:")
                    print(final_response)
                    print(f"{'='*60}")

                return final_response

            else:
                # 既没有工具调用也没有文本内容
                if choice.finish_reason == "stop":
                    return "任务已完成。"
                else:
                    print(f"[Agent] 意外的响应: finish_reason={choice.finish_reason}")
                    return "Agent 响应异常"

        # 超过最大轮次
        return "达到最大执行轮次限制，任务可能未完全完成。"

    def reset(self):
        """重置对话历史"""
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        print("[Agent] 对话历史已重置")

    def get_history(self) -> list:
        """获取对话历史"""
        return self.messages.copy()
