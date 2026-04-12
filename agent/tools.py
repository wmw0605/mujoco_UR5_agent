"""
Agent 工具定义
定义 LLM Agent 可调用的工具（OpenAI Function Calling 格式）
"""

import json
from robot.base import RobotInterface


# ==================== 工具 JSON Schema 定义 ====================

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_scene_info",
            "description": "获取当前场景信息，包括桌面上所有物体的名称、描述、位置，以及机械臂当前状态。这是规划动作的第一步。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_robot_state",
            "description": "获取机械臂的详细状态信息，包括TCP位姿、关节角度和夹爪开合状态。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_to",
            "description": "移动机械臂末端执行器到指定的三维坐标位置。坐标单位为米。注意：移动前确保目标位置在安全工作范围内。",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "目标 X 坐标（米），X正方向=远离基座。工作范围约 0.15~0.85",
                    },
                    "y": {
                        "type": "number",
                        "description": "目标 Y 坐标（米），Y正方向=左侧。工作范围约 -0.5~0.5",
                    },
                    "z": {
                        "type": "number",
                        "description": "目标 Z 坐标（米），Z正方向=向上。桌面高度0.37m，安全高度>=0.5m",
                    },
                },
                "required": ["x", "y", "z"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grasp",
            "description": "闭合夹爪抓取物体。在调用前，确保末端执行器已移动到物体的抓取位置。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "release",
            "description": "打开夹爪释放当前抓持的物体。在调用前，确保末端执行器已移动到目标放置位置。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "go_home",
            "description": "将机械臂移动回初始位姿（Home位置）。在完成任务或需要重新开始时使用。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pick_and_place",
            "description": "高级工具：自动完成「抓取指定物体 → 放置到目标位置」的完整流程。包含移动到物体上方、下降抓取、提升、移动到目标、下降放置、释放等步骤。",
            "parameters": {
                "type": "object",
                "properties": {
                    "object_name": {
                        "type": "string",
                        "description": "要抓取的物体名称，例如 'red_cube'、'green_cube'、'blue_cylinder'、'yellow_cube'",
                    },
                    "target_x": {
                        "type": "number",
                        "description": "目标放置位置的 X 坐标（米）",
                    },
                    "target_y": {
                        "type": "number",
                        "description": "目标放置位置的 Y 坐标（米）",
                    },
                    "target_z": {
                        "type": "number",
                        "description": "目标放置位置的 Z 坐标（米），通常为桌面高度 0.39m",
                    },
                },
                "required": ["object_name", "target_x", "target_y", "target_z"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pick_object",
            "description": "单独抓取并提起一个指定的物体，抓起后保持在抓取点上方悬空。如果只需要抓起来，调用此工具即可。",
            "parameters": {
                "type": "object",
                "properties": {
                    "object_name": {
                        "type": "string",
                        "description": "要抓取的物体名称，例如 'red_cube'、'green_cube'、'blue_cylinder'、'yellow_cube'",
                    },
                },
                "required": ["object_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place_object",
            "description": "将当前抓取的物体放置到指定的三维坐标位置。配合 pick_object 使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "目标放置位置的 X 坐标（米）",
                    },
                    "y": {
                        "type": "number",
                        "description": "目标放置位置的 Y 坐标（米）",
                    },
                    "z": {
                        "type": "number",
                        "description": "目标放置位置的 Z 坐标（米），通常为桌面高度 0.39m",
                    },
                },
                "required": ["x", "y", "z"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scan_workspace",
            "description": "使用腕部 RGB-D 相机扫描工作空间并检测物体。机械臂将自动移动到桌面上方的扫描位置，捕获彩色图像和深度图，通过视觉算法检测物体并计算其 3D 世界坐标。这是获取物体位置的主要方式，每次操作前必须先调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


# ==================== 工具执行函数 ====================

def execute_tool(tool_name: str, arguments: dict, robot: RobotInterface) -> str:
    """执行指定工具并返回结果

    Args:
        tool_name: 工具名称
        arguments: 工具参数字典
        robot: 机器人接口实例

    Returns:
        工具执行结果的 JSON 字符串
    """
    try:
        if tool_name == "get_scene_info":
            result = robot.get_scene_info()
            return result  # 直接返回文本描述

        elif tool_name == "get_robot_state":
            state = robot.get_robot_state()
            return json.dumps(state, ensure_ascii=False, indent=2)

        elif tool_name == "move_to":
            x = float(arguments["x"])
            y = float(arguments["y"])
            z = float(arguments["z"])
            result = robot.move_to_position(x, y, z)
            return json.dumps(result, ensure_ascii=False)

        elif tool_name == "grasp":
            result = robot.close_gripper()
            return json.dumps(result, ensure_ascii=False)

        elif tool_name == "release":
            result = robot.open_gripper()
            return json.dumps(result, ensure_ascii=False)

        elif tool_name == "go_home":
            result = robot.go_home()
            return json.dumps(result, ensure_ascii=False)

        elif tool_name == "pick_and_place":
            object_name = arguments["object_name"]
            target_x = float(arguments["target_x"])
            target_y = float(arguments["target_y"])
            target_z = float(arguments["target_z"])

            # 先抓取
            pick_result = robot.pick_object(object_name)
            if not pick_result["success"]:
                return json.dumps(pick_result, ensure_ascii=False)

            # 再放置
            place_result = robot.place_object(target_x, target_y, target_z)
            combined = {
                "success": place_result["success"],
                "message": f"抓取: {pick_result['message']}; 放置: {place_result['message']}",
            }
            return json.dumps(combined, ensure_ascii=False)

        elif tool_name == "pick_object":
            object_name = arguments["object_name"]
            result = robot.pick_object(object_name)
            return json.dumps(result, ensure_ascii=False)

        elif tool_name == "place_object":
            x = float(arguments["x"])
            y = float(arguments["y"])
            z = float(arguments["z"])
            result = robot.place_object(x, y, z)
            return json.dumps(result, ensure_ascii=False)

        elif tool_name == "scan_workspace":
            result = robot.scan_workspace()
            # 返回场景描述文本（包含所有检测到的物体信息）
            return result.get("description", json.dumps(result, ensure_ascii=False))

        else:
            return json.dumps({
                "success": False,
                "message": f"未知工具: {tool_name}",
            }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "success": False,
            "message": f"工具执行异常: {str(e)}",
        }, ensure_ascii=False)
