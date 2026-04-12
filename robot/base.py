"""
机器人接口抽象基类
定义仿真和真实机器人的统一接口
"""

from abc import ABC, abstractmethod
import numpy as np


class RobotInterface(ABC):
    """机器人接口抽象基类

    所有机器人实现（仿真/真实）都必须继承此类，
    确保 Agent 的工具调用可以无缝切换到不同的机器人后端。
    """

    @abstractmethod
    def get_tcp_pose(self) -> dict:
        """获取 TCP（工具中心点）位姿

        Returns:
            {"position": [x, y, z], "orientation": [rx, ry, rz]}
        """
        pass

    @abstractmethod
    def get_joint_positions(self) -> list[float]:
        """获取当前关节角度（弧度）

        Returns:
            6 个关节角度列表
        """
        pass

    @abstractmethod
    def get_gripper_state(self) -> dict:
        """获取夹爪状态

        Returns:
            {"is_open": bool, "openness": float (0-1)}
        """
        pass

    @abstractmethod
    def move_to_position(
        self,
        x: float, y: float, z: float,
        rx: float = None, ry: float = None, rz: float = None,
        speed: float = 0.5,
    ) -> dict:
        """移动末端执行器到目标位置

        Args:
            x, y, z: 目标位置（米）
            rx, ry, rz: 目标姿态（弧度），None 则保持当前姿态
            speed: 速度因子 0-1

        Returns:
            {"success": bool, "message": str, "final_position": [x,y,z]}
        """
        pass

    @abstractmethod
    def open_gripper(self) -> dict:
        """打开夹爪

        Returns:
            {"success": bool, "message": str}
        """
        pass

    @abstractmethod
    def close_gripper(self) -> dict:
        """闭合夹爪

        Returns:
            {"success": bool, "message": str}
        """
        pass

    @abstractmethod
    def go_home(self) -> dict:
        """回到初始位姿

        Returns:
            {"success": bool, "message": str}
        """
        pass

    @abstractmethod
    def get_scene_info(self) -> str:
        """获取场景信息（文本描述）

        Returns:
            场景描述字符串
        """
        pass

    @abstractmethod
    def pick_object(self, object_name: str) -> dict:
        """抓取指定物体

        Args:
            object_name: 物体名称

        Returns:
            {"success": bool, "message": str}
        """
        pass

    @abstractmethod
    def place_object(self, x: float, y: float, z: float) -> dict:
        """放置物体到指定位置

        Args:
            x, y, z: 目标位置

        Returns:
            {"success": bool, "message": str}
        """
        pass

    @abstractmethod
    def scan_workspace(self) -> dict:
        """使用腕部相机扫描工作空间并检测物体

        机械臂将移动到扫描位姿，使用腕部相机捕获 RGB-D 图像，
        通过视觉算法检测物体并返回 3D 世界坐标。

        Returns:
            {
                "success": bool,
                "message": str,
                "objects": list[dict],  # 检测到的物体列表
                "description": str,     # 场景文本描述
            }
        """
        pass
