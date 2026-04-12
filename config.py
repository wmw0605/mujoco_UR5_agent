"""
全局配置
"""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    """系统配置"""

    # ==================== 运行模式 ====================
    mode: str = "sim"  # "sim" (仿真) 或 "real" (真实机器人)

    # ==================== MuJoCo 仿真配置 ====================
    model_path: str = "models/scene.xml"
    render: bool = True  # 是否启用 GUI 可视化
    camera_width: int = 640
    camera_height: int = 480

    # ==================== Qwen Agent 配置 ====================
    qwen_model: str = "qwen-max"  # qwen-max / qwen-plus / qwen-turbo
    dashscope_api_key: str = "sk-f1c8b422da5d4582b78a5ad5e0e7557b"
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # ==================== 真实机器人配置 ====================
    robot_ip: str = "192.168.1.100"
    gripper_port: str = "/dev/ttyUSB0"

    def __post_init__(self):
        """从环境变量补充配置"""
        if not self.dashscope_api_key:
            self.dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "")

    @classmethod
    def from_args(cls, args) -> "Config":
        """从命令行参数创建配置"""
        config = cls()
        if hasattr(args, "mode") and args.mode:
            config.mode = args.mode
        if hasattr(args, "model") and args.model:
            config.qwen_model = args.model
        if hasattr(args, "no_render") and args.no_render:
            config.render = False
        if hasattr(args, "robot_ip") and args.robot_ip:
            config.robot_ip = args.robot_ip
        return config
