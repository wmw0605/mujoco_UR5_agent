"""
仿真机器人实现
基于 SimEnvironment + RobotController + SimCamera 的仿真机器人
"""

import numpy as np
from scipy.spatial.transform import Rotation

from robot.base import RobotInterface
from sim.environment import SimEnvironment
from sim.camera import SimCamera
from sim.controller import RobotController


class SimRobot(RobotInterface):
    """仿真模式下的机器人实现

    封装 MuJoCo 仿真环境，提供与 RobotInterface 一致的接口。
    """

    def __init__(self, model_path: str = "models/scene.xml", render: bool = True):
        """初始化仿真机器人

        Args:
            model_path: MuJoCo 模型路径
            render: 是否启用 GUI
        """
        # 初始化仿真环境
        self.env = SimEnvironment(model_path=model_path, render=render)
        # 初始化控制器
        self.controller = RobotController(self.env)
        # 初始化相机
        self.camera = SimCamera(self.env)

        print("[SimRobot] 仿真机器人初始化完成")

    def get_tcp_pose(self) -> dict:
        """获取 TCP 位姿"""
        pos, rot_mat = self.env.get_ee_pose()
        # 旋转矩阵转欧拉角 (rx, ry, rz)
        euler = Rotation.from_matrix(rot_mat).as_euler('xyz')
        return {
            "position": pos.tolist(),
            "orientation": euler.tolist(),
        }

    def get_joint_positions(self) -> list[float]:
        """获取关节角度"""
        return self.env.get_joint_positions().tolist()

    def get_gripper_state(self) -> dict:
        """获取夹爪状态"""
        openness = self.env.get_gripper_state()
        return {
            "is_open": bool(openness < 0.3),
            "openness": round(1.0 - openness, 2),  # 反转：1=全开, 0=全闭
        }

    def move_to_position(
        self,
        x: float, y: float, z: float,
        rx: float = None, ry: float = None, rz: float = None,
        speed: float = 0.5,
    ) -> dict:
        """移动到目标位置"""
        target_pos = np.array([x, y, z])

        # 如果提供了姿态
        target_quat = None
        if rx is not None and ry is not None and rz is not None:
            target_quat = Rotation.from_euler('xyz', [rx, ry, rz]).as_quat()
            # scipy 的四元数是 (x,y,z,w)，MuJoCo 是 (w,x,y,z)
            target_quat = np.array([target_quat[3], target_quat[0], target_quat[1], target_quat[2]])

        success = bool(self.controller.move_to_pose(target_pos, target_quat, speed))

        final_pos, _ = self.env.get_ee_pose()
        return {
            "success": success,
            "message": f"已移动到 ({final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f})" if success
                       else "移动失败，目标位置可能不可达",
            "final_position": final_pos.tolist(),
        }

    def open_gripper(self) -> dict:
        """打开夹爪"""
        success = bool(self.controller.open_gripper())
        return {
            "success": success,
            "message": "夹爪已打开" if success else "夹爪打开失败",
        }

    def close_gripper(self) -> dict:
        """闭合夹爪"""
        success = bool(self.controller.close_gripper())
        state = self.env.get_gripper_state()
        grasped = bool(0.1 < state < 0.9)
        return {
            "success": success,
            "message": "夹爪已闭合，已抓住物体" if grasped else "夹爪已闭合",
            "object_grasped": grasped,
        }

    def go_home(self) -> dict:
        """回到初始位姿"""
        success = bool(self.controller.go_home())
        return {
            "success": success,
            "message": "已回到初始位姿" if success else "回到初始位姿失败",
        }

    def get_scene_info(self) -> str:
        """获取场景信息"""
        return self.camera.get_scene_description()

    def pick_object(self, object_name: str) -> dict:
        """抓取指定物体"""
        # 查找物体位置
        try:
            pos, _ = self.env.get_body_pose(object_name)
        except ValueError:
            return {
                "success": False,
                "message": f"未找到物体: {object_name}",
            }

        # 调整抓取位置（直接使用物体中心，controller.pick 会处理 pinch 偏移）
        grasp_pos = pos.copy()

        success = bool(self.controller.pick(grasp_pos))
        desc = self.env.OBJECT_DESCRIPTIONS.get(object_name, object_name)

        return {
            "success": success,
            "message": f"已成功抓取 {desc}" if success else f"抓取 {desc} 失败",
        }

    def place_object(self, x: float, y: float, z: float) -> dict:
        """放置物体"""
        target_pos = np.array([x, y, z])
        success = bool(self.controller.place(target_pos))

        return {
            "success": success,
            "message": f"已放置到 ({x:.3f}, {y:.3f}, {z:.3f})" if success
                       else "放置失败",
        }

    def capture_image(self, camera_name: str = "overhead_cam") -> np.ndarray:
        """捕获相机图像

        Args:
            camera_name: 相机名称

        Returns:
            RGB 图像
        """
        return self.camera.capture_rgb(camera_name)

    def scan_workspace(self) -> dict:
        """使用腕部相机扫描工作空间并检测物体

        采用多位置扫描策略，确保覆盖整个桌面并避免机械臂自遮挡：
        1. 打开夹爪避免遮挡
        2. 依次移到多个扫描位姿（中心、左前、右前）
        3. 每个位置进行视觉检测
        4. 合并去重所有检测结果
        5. 保存最终检测可视化

        Returns:
            {"success": bool, "message": str, "objects": list, "description": str}
        """
        print("[SimRobot] 开始多位置腕部相机扫描...")

        # 1. 打开夹爪避免遮挡视野
        self.controller.open_gripper()

        # 2. 定义多个扫描位姿（覆盖桌面不同区域）
        scan_quat = np.array([0.0, 1.0, 0.0, 0.0])  # 竖直向下
        scan_positions = [
            np.array([0.35, 0.0, 0.70]),    # 中心
            np.array([0.50, 0.15, 0.70]),   # 左前
            np.array([0.50, -0.15, 0.70]),  # 右前
        ]

        all_detected = {}  # name -> best detection (highest confidence)
        last_rgb = None

        for i, scan_pos in enumerate(scan_positions):
            print(f"[SimRobot] 扫描位置 {i+1}/{len(scan_positions)}: {scan_pos}")
            success = bool(self.controller.move_to_pose(scan_pos, scan_quat))
            if not success:
                print(f"[SimRobot] 无法到达扫描位置 {i+1}，跳过")
                continue

            # 等待稳定
            self.env.step(300)

            # 视觉检测
            detected, rgb, depth = self.camera.detect_objects_by_vision("wrist_cam")
            last_rgb = rgb

            # 合并结果（保留置信度最高的检测）
            for obj in detected:
                name = obj["name"]
                if name not in all_detected or obj.get("confidence", 0) > all_detected[name].get("confidence", 0):
                    all_detected[name] = obj

        # 3. 汇总结果
        final_detected = list(all_detected.values())

        # 4. 保存最后一次扫描的检测可视化
        from pathlib import Path
        output_dir = Path(__file__).parent.parent
        if last_rgb is not None:
            self.camera.save_detection_image(
                last_rgb, final_detected, str(output_dir / "detection_result.png")
            )

        # 5. 生成场景描述
        description = self.camera.get_vision_scene_description(
            final_detected, "wrist_cam"
        )

        # 缓存最新的检测结果
        self._last_scan_result = final_detected

        print(f"[SimRobot] 扫描完成，共检测到 {len(final_detected)} 个物体")

        return {
            "success": True,
            "message": f"扫描完成，检测到 {len(final_detected)} 个物体",
            "objects": final_detected,
            "description": description,
        }

    def get_robot_state(self) -> dict:
        """获取完整的机器人状态"""
        tcp = self.get_tcp_pose()
        joints = self.get_joint_positions()
        gripper = self.get_gripper_state()

        return {
            "tcp_position": tcp["position"],
            "tcp_orientation": tcp["orientation"],
            "joint_positions": joints,
            "gripper_is_open": gripper["is_open"],
            "gripper_openness": gripper["openness"],
        }

    def is_running(self) -> bool:
        """检查仿真是否仍在运行"""
        return self.env.is_viewer_running()

    def close(self):
        """关闭仿真"""
        self.camera.close()
        self.env.close()
