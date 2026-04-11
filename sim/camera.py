"""
仿真相机模块
提供 RGB/Depth 图像渲染和场景感知功能
"""

import mujoco
import numpy as np
import cv2
from pathlib import Path


class SimCamera:
    """仿真相机 - 用于场景感知

    支持 RGB 和深度图渲染，以及基于仿真真值的物体检测。
    """

    def __init__(self, env, width: int = 640, height: int = 480):
        """初始化相机

        Args:
            env: SimEnvironment 实例
            width: 图像宽度
            height: 图像高度
        """
        self.env = env
        self.width = width
        self.height = height

        # 创建离屏渲染器
        self.renderer = mujoco.Renderer(env.model, height=height, width=width)
        print(f"[Camera] 渲染器初始化完成 ({width}x{height})")

    def capture_rgb(self, camera_name: str = "overhead_cam") -> np.ndarray:
        """捕获 RGB 图像

        Args:
            camera_name: 相机名称 ("overhead_cam" 或 "wrist_cam")

        Returns:
            RGB 图像 (H, W, 3) uint8
        """
        self.renderer.update_scene(self.env.data, camera=camera_name)
        rgb = self.renderer.render()
        return rgb.copy()

    def capture_depth(self, camera_name: str = "overhead_cam") -> np.ndarray:
        """捕获深度图

        Args:
            camera_name: 相机名称

        Returns:
            深度图 (H, W) float32，单位：米
        """
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.env.data, camera=camera_name)
        depth = self.renderer.render()
        self.renderer.disable_depth_rendering()
        return depth.copy()

    def capture_rgbd(self, camera_name: str = "overhead_cam") -> tuple[np.ndarray, np.ndarray]:
        """同时捕获 RGB 和深度图

        Args:
            camera_name: 相机名称

        Returns:
            (rgb (H,W,3), depth (H,W))
        """
        rgb = self.capture_rgb(camera_name)
        depth = self.capture_depth(camera_name)
        return rgb, depth

    def detect_objects(self) -> list[dict]:
        """基于仿真真值的物体检测

        直接从 MjData 获取物体位姿信息。
        在真实环境中，此方法应替换为视觉检测模型。

        Returns:
            物体列表，每个物体包含:
            - name: 物体标识名
            - description: 中文描述
            - position: [x, y, z] 世界坐标
            - on_table: 是否在桌面上
        """
        objects = []
        all_poses = self.env.get_all_object_poses()

        for name, info in all_poses.items():
            pos = info["pos"]
            # 判断物体是否在桌面上（z > 0.35 且在桌面范围内）
            on_table = (
                pos[2] > 0.35
                and 0.15 < pos[0] < 0.85
                and -0.5 < pos[1] < 0.5
            )

            objects.append({
                "name": name,
                "description": info["description"],
                "position": pos.tolist(),
                "on_table": on_table,
            })

        return objects

    def get_scene_description(self) -> str:
        """生成场景的文本描述，供 LLM Agent 使用

        Returns:
            场景的自然语言描述
        """
        objects = self.detect_objects()

        # 获取机械臂状态
        ee_pos, _ = self.env.get_ee_pose()
        gripper_state = self.env.get_gripper_state()
        gripper_status = "闭合" if gripper_state > 0.5 else "张开"

        # 构建描述
        lines = []
        lines.append("=== 当前场景信息 ===")
        lines.append(f"\n机械臂末端位置: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})")
        lines.append(f"夹爪状态: {gripper_status} (开合度: {gripper_state:.2f})")
        lines.append(f"\n桌面高度: 0.37m")
        lines.append(f"桌面范围: X[0.15, 0.85], Y[-0.50, 0.50]")
        lines.append(f"\n物体列表:")

        for obj in objects:
            pos = obj["position"]
            status = "在桌面上" if obj["on_table"] else "不在桌面上"
            lines.append(
                f"  - {obj['description']} ({obj['name']}): "
                f"位置 ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), {status}"
            )

        lines.append(f"\n坐标系说明: X正方向=远离机器人基座, Y正方向=左侧, Z正方向=向上")
        lines.append(f"安全抓取高度: Z=0.50m (接近时先到此高度)")
        lines.append(f"抓取高度: Z=0.39m (与桌面齐平)")

        return "\n".join(lines)

    def save_image(self, image: np.ndarray, filepath: str):
        """保存图像到文件

        Args:
            image: RGB 图像数组
            filepath: 保存路径
        """
        # OpenCV 使用 BGR 格式
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, bgr)
        print(f"[Camera] 图像已保存: {filepath}")

    def close(self):
        """释放渲染器资源"""
        if hasattr(self, 'renderer') and self.renderer is not None:
            self.renderer.close()
            self.renderer = None
