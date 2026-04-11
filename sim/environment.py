"""
MuJoCo 仿真环境封装
提供 UR5e + Robotiq 2F-85 的物理仿真环境管理
"""

import mujoco
import mujoco.viewer
import numpy as np
import threading
import time
from pathlib import Path


class SimEnvironment:
    """MuJoCo 仿真环境

    管理模型加载、物理仿真步进、GUI 可视化和状态查询。
    """

    # UR5e 的 6 个关节名称
    ARM_JOINTS = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    # UR5e 的 6 个执行器名称
    ARM_ACTUATORS = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
    ]

    # Robotiq 夹爪执行器
    GRIPPER_ACTUATOR = "fingers_actuator"

    # 可抓取物体
    OBJECT_NAMES = ["red_cube", "green_cube", "blue_cylinder", "yellow_cube"]

    # 物体描述映射
    OBJECT_DESCRIPTIONS = {
        "red_cube": "红色方块",
        "green_cube": "绿色方块",
        "blue_cylinder": "蓝色圆柱",
        "yellow_cube": "黄色方块",
    }

    def __init__(self, model_path: str = "models/scene.xml", render: bool = True):
        """初始化仿真环境

        Args:
            model_path: MuJoCo 模型文件路径
            render: 是否启用 GUI 渲染
        """
        # 解析模型路径
        self.model_path = Path(model_path)
        if not self.model_path.is_absolute():
            # 相对于项目根目录
            self.model_path = Path(__file__).parent.parent / self.model_path

        # 加载模型
        print(f"[SimEnv] 加载模型: {self.model_path}")
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

        # 设置仿真参数
        self.model.opt.timestep = 0.002  # 2ms 时间步

        # 获取执行器索引
        self._arm_actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in self.ARM_ACTUATORS
        ]
        self._gripper_actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.GRIPPER_ACTUATOR
        )

        # 获取关节索引
        self._arm_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.ARM_JOINTS
        ]

        # 获取末端执行器 site 索引
        self._ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )
        self._pinch_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch"
        )

        # 重置到初始关键帧
        self.reset()

        # 启动 GUI viewer
        self.render_enabled = render
        self.viewer = None
        self._viewer_thread = None
        if render:
            self._start_viewer()

        print("[SimEnv] 仿真环境初始化完成")

    def _start_viewer(self):
        """启动 MuJoCo GUI viewer（在主线程或后台线程）"""
        try:
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data, show_left_ui=True, show_right_ui=True
            )
            print("[SimEnv] GUI Viewer 已启动")
        except Exception as e:
            print(f"[SimEnv] 警告: 无法启动 GUI Viewer: {e}")
            print("[SimEnv] 将以无头模式运行")
            self.render_enabled = False
            self.viewer = None

    def reset(self):
        """重置仿真到初始状态"""
        # 查找 "home" 关键帧
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)

        # 前向运动学计算
        mujoco.mj_forward(self.model, self.data)
        print("[SimEnv] 仿真已重置到初始状态")

    def step(self, n_steps: int = 1):
        """推进仿真 n 步

        Args:
            n_steps: 仿真步数
        """
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

    def step_until_stable(self, max_steps: int = 2000, vel_threshold: float = 0.01):
        """仿真直到系统稳定

        Args:
            max_steps: 最大仿真步数
            vel_threshold: 速度阈值，低于此值认为稳定
        """
        for i in range(max_steps):
            mujoco.mj_step(self.model, self.data)
            if i % 50 == 0 and self.viewer is not None and self.viewer.is_running():
                self.viewer.sync()
            # 检查关节速度是否足够小
            arm_vel = np.abs(self.get_joint_velocities())
            if np.max(arm_vel) < vel_threshold:
                break

        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

    def set_arm_ctrl(self, ctrl: np.ndarray):
        """设置机械臂控制输入

        Args:
            ctrl: 6 维控制数组（关节目标位置）
        """
        for i, act_id in enumerate(self._arm_actuator_ids):
            self.data.ctrl[act_id] = ctrl[i]

    def set_gripper_ctrl(self, value: float):
        """设置夹爪控制输入

        Args:
            value: 0 = 全开, 255 = 全闭
        """
        self.data.ctrl[self._gripper_actuator_id] = np.clip(value, 0, 255)

    def get_joint_positions(self) -> np.ndarray:
        """获取机械臂关节角度 (6,)"""
        positions = np.zeros(6)
        for i, joint_id in enumerate(self._arm_joint_ids):
            qpos_addr = self.model.jnt_qposadr[joint_id]
            positions[i] = self.data.qpos[qpos_addr]
        return positions

    def get_joint_velocities(self) -> np.ndarray:
        """获取机械臂关节速度 (6,)"""
        velocities = np.zeros(6)
        for i, joint_id in enumerate(self._arm_joint_ids):
            qvel_addr = self.model.jnt_dofadr[joint_id]
            velocities[i] = self.data.qvel[qvel_addr]
        return velocities

    def get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """获取末端执行器位姿

        Returns:
            (position (3,), rotation_matrix (3,3))
        """
        pos = self.data.site_xpos[self._ee_site_id].copy()
        rot = self.data.site_xmat[self._ee_site_id].copy().reshape(3, 3)
        return pos, rot

    def get_pinch_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """获取夹爪指尖中心位姿

        Returns:
            (position (3,), rotation_matrix (3,3))
        """
        pos = self.data.site_xpos[self._pinch_site_id].copy()
        rot = self.data.site_xmat[self._pinch_site_id].copy().reshape(3, 3)
        return pos, rot

    def get_gripper_state(self) -> float:
        """获取夹爪状态

        Returns:
            夹爪开合值 0.0(全开) ~ 1.0(全闭)
        """
        # 读取 right_driver_joint 的位置
        joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_driver_joint"
        )
        qpos_addr = self.model.jnt_qposadr[joint_id]
        # 关节范围 0~0.8，归一化到 0~1
        return float(np.clip(self.data.qpos[qpos_addr] / 0.8, 0.0, 1.0))

    def get_body_pose(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        """获取物体的世界位姿

        Args:
            body_name: 物体名称

        Returns:
            (position (3,), quaternion (4,))
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"未找到物体: {body_name}")
        pos = self.data.xpos[body_id].copy()
        quat = self.data.xquat[body_id].copy()
        return pos, quat

    def get_all_object_poses(self) -> dict:
        """获取所有可抓取物体的位姿

        Returns:
            {物体名: {"pos": (3,), "quat": (4,), "description": str}}
        """
        objects = {}
        for name in self.OBJECT_NAMES:
            try:
                pos, quat = self.get_body_pose(name)
                objects[name] = {
                    "pos": pos,
                    "quat": quat,
                    "description": self.OBJECT_DESCRIPTIONS.get(name, name),
                }
            except ValueError:
                continue
        return objects

    def is_viewer_running(self) -> bool:
        """检查 GUI viewer 是否仍在运行"""
        if self.viewer is None:
            return False
        return self.viewer.is_running()

    def close(self):
        """关闭仿真环境"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        print("[SimEnv] 仿真环境已关闭")
