"""
运动控制器模块
提供 IK 求解、轨迹生成和夹爪控制
"""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import time


class RobotController:
    """机器人运动控制器

    基于 Jacobian 伪逆的 IK 求解，以及轨迹插值控制。
    """

    # 默认 home 关节角（UR5e 标准 home）
    HOME_QPOS = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])

    # 竖直向下的末端姿态四元数 (w, x, y, z) —— attachment_site Z轴朝下
    GRASP_QUAT = np.array([0.0, 1.0, 0.0, 0.0])

    # 实际上 pinch site 在 attachment_site 下方约 0.1488m
    # 考虑少许余量和执行器误差，使用 0.140m
    GRASP_OFFSET_Z = 0.140

    def __init__(self, env):
        """初始化控制器

        Args:
            env: SimEnvironment 实例
        """
        self.env = env
        self.model = env.model
        self.data = env.data

        # IK 参数
        self.ik_max_iterations = 500
        self.ik_tolerance = 1e-3  # 位置精度 1mm
        self.ik_damping = 1e-4  # 阻尼系数（防止奇异位形）
        self.ik_step_size = 0.8  # 步长因子

        print("[Controller] 运动控制器初始化完成")

    def inverse_kinematics(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray = None,
        init_qpos: np.ndarray = None,
    ) -> np.ndarray:
        """使用 Jacobian 伪逆法求解逆运动学

        Args:
            target_pos: 目标位置 (3,)
            target_quat: 目标四元数 (4,)，None 则只考虑位置
            init_qpos: 初始关节角 (6,)，None 则使用当前关节角

        Returns:
            目标关节角度 (6,)

        Raises:
            RuntimeError: IK 求解失败
        """
        # 使用临时 data 进行 IK 计算，不影响仿真状态
        ik_data = mujoco.MjData(self.model)
        # 复制当前仿真状态（包括freejoint的正确初始化）
        ik_data.qpos[:] = self.data.qpos[:]
        ik_data.qvel[:] = 0

        # 设置初始关节角
        if init_qpos is not None:
            q = init_qpos.copy()
        else:
            q = self.env.get_joint_positions()

        # 写入关节角到临时 data
        for i, joint_id in enumerate(self.env._arm_joint_ids):
            qpos_addr = self.model.jnt_qposadr[joint_id]
            ik_data.qpos[qpos_addr] = q[i]

        site_id = self.env._ee_site_id
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))

        # 获取 arm joint 的 dof 地址
        dof_ids = []
        for joint_id in self.env._arm_joint_ids:
            dof_ids.append(self.model.jnt_dofadr[joint_id])
        dof_ids = np.array(dof_ids)

        for iteration in range(self.ik_max_iterations):
            # 前向运动学
            mujoco.mj_forward(self.model, ik_data)

            # 当前末端位置
            current_pos = ik_data.site_xpos[site_id].copy()

            # 位置误差
            pos_error = target_pos - current_pos

            # 检查是否收敛
            pos_norm = np.linalg.norm(pos_error)
            if pos_norm < self.ik_tolerance:
                if target_quat is None:
                    return q.copy()

            # 如果同时有姿态约束
            if target_quat is not None:
                # 计算姿态误差
                current_mat = ik_data.site_xmat[site_id].copy().reshape(3, 3)
                target_mat = np.zeros(9)
                mujoco.mju_quat2Mat(target_mat, target_quat)
                target_mat = target_mat.reshape(3, 3)

                # 使用旋转矩阵差来计算姿态误差
                error_mat = target_mat @ current_mat.T
                error_rotvec = Rotation.from_matrix(error_mat).as_rotvec()

                rot_norm = np.linalg.norm(error_rotvec)
                if pos_norm < self.ik_tolerance and rot_norm < self.ik_tolerance * 10:
                    return q.copy()

                # 6D 误差向量（降低姿态权重，优先保证位置精度）
                error = np.concatenate([pos_error, error_rotvec * 0.3])

                # 计算完整 Jacobian
                mujoco.mj_jacSite(self.model, ik_data, jacp, jacr, site_id)
                jac = np.vstack([jacp[:, dof_ids], jacr[:, dof_ids]])
            else:
                # 仅位置 3D 误差
                error = pos_error

                # 仅位置 Jacobian
                mujoco.mj_jacSite(self.model, ik_data, jacp, jacr, site_id)
                jac = jacp[:, dof_ids]

            # 阻尼最小二乘法 (DLS)
            n = jac.shape[1]
            jac_T = jac.T
            dq = jac_T @ np.linalg.solve(
                jac @ jac_T + self.ik_damping * np.eye(jac.shape[0]),
                error
            )

            # 更新关节角
            q += self.ik_step_size * dq

            # 关节角限幅并归一化到 [-pi, pi]
            for i, joint_id in enumerate(self.env._arm_joint_ids):
                # 先归一化到 [-pi, pi]
                q[i] = np.mod(q[i] + np.pi, 2 * np.pi) - np.pi
                # 再限幅
                q[i] = np.clip(
                    q[i],
                    self.model.jnt_range[joint_id, 0],
                    self.model.jnt_range[joint_id, 1],
                )

            # 写回临时 data
            for i, joint_id in enumerate(self.env._arm_joint_ids):
                qpos_addr = self.model.jnt_qposadr[joint_id]
                ik_data.qpos[qpos_addr] = q[i]

        # 最后检查是否足够接近
        mujoco.mj_forward(self.model, ik_data)
        final_pos = ik_data.site_xpos[site_id].copy()
        final_error = np.linalg.norm(target_pos - final_pos)

        if final_error > self.ik_tolerance * 50:
            raise RuntimeError(
                f"IK 求解未收敛: 目标 {target_pos}, 当前 {final_pos}, "
                f"误差 {final_error:.4f}m"
            )

        print(f"[Controller] IK 求解完成: 误差 {final_error:.4f}m, 迭代 {iteration+1} 次")

        # 将关节角归一化到 [-pi, pi] 范围，避免多圈旋转
        q = np.mod(q + np.pi, 2 * np.pi) - np.pi

        return q

    def move_to_pose(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray = None,
        speed: float = 0.5,
        use_ik: bool = True,
    ) -> bool:
        """移动末端执行器到目标位姿

        Args:
            target_pos: 目标位置 (3,)
            target_quat: 目标四元数 (4,)，None 则只考虑位置
            speed: 速度因子 0.0~1.0
            use_ik: 是否使用 IK（False 则需要直接提供关节角）

        Returns:
            是否成功到达
        """
        target_pos = np.asarray(target_pos, dtype=float)

        if use_ik:
            try:
                target_joints = self.inverse_kinematics(target_pos, target_quat)
            except RuntimeError as e:
                print(f"[Controller] 运动失败: {e}")
                return False
        else:
            target_joints = target_pos  # 直接作为关节角

        return self.move_to_joints(target_joints, speed)

    def move_to_joints(self, target_joints: np.ndarray, speed: float = 0.5) -> bool:
        """关节空间运动

        使用线性插值平滑移动到目标关节角。

        Args:
            target_joints: 目标关节角 (6,)
            speed: 速度因子 0.0~1.0

        Returns:
            是否成功到达
        """
        target_joints = np.asarray(target_joints, dtype=float)
        current_joints = self.env.get_joint_positions()

        # 计算最大关节变化量，决定插值步数
        max_diff = np.max(np.abs(target_joints - current_joints))
        n_steps = max(int(max_diff / (0.001 * speed)), 200)  # 足够多步保证平滑

        # 线性插值轨迹
        for i in range(n_steps + 1):
            alpha = i / n_steps
            # 使用平滑插值（S 曲线）
            alpha_smooth = 3 * alpha**2 - 2 * alpha**3
            interp_joints = current_joints + alpha_smooth * (target_joints - current_joints)

            # 设置控制目标
            self.env.set_arm_ctrl(interp_joints)

            # 仿真推进（每帧10步，给执行器更多跟踪时间）
            self.env.step(10)

        # 到达后持续保持目标一段时间，让执行器完全收敛
        self.env.set_arm_ctrl(target_joints)
        self.env.step(2000)

        # 等待稳定
        self.env.step_until_stable(max_steps=3000)

        # 检查到达精度
        final_joints = self.env.get_joint_positions()
        error = np.max(np.abs(final_joints - target_joints))
        success = error < 0.2  # 约 11.5 度

        if success:
            print(f"[Controller] 关节运动完成，最大误差: {error:.4f} rad")
        else:
            print(f"[Controller] 关节运动精度不足，最大误差: {error:.4f} rad")

        return success

    def open_gripper(self, speed: float = 0.5) -> bool:
        """打开夹爪

        Args:
            speed: 速度因子

        Returns:
            是否成功
        """
        print("[Controller] 打开夹爪...")
        self.env.set_gripper_ctrl(0)  # 0 = 全开

        # 等待夹爪运动完成
        n_steps = int(500 / speed)
        self.env.step(n_steps)

        state = self.env.get_gripper_state()
        print(f"[Controller] 夹爪已打开 (开合度: {state:.2f})")
        return True

    def close_gripper(self, force: float = 255, speed: float = 0.5) -> bool:
        """闭合夹爪

        Args:
            force: 闭合力度 (0-255)
            speed: 速度因子

        Returns:
            是否成功抓住物体
        """
        print("[Controller] 闭合夹爪...")
        self.env.set_gripper_ctrl(force)

        # 等待夹爪运动完成
        n_steps = int(800 / speed)
        self.env.step(n_steps)

        state = self.env.get_gripper_state()
        # 如果夹爪没有完全闭合，说明可能夹住了物体
        grasped = 0.1 < state < 0.9
        if grasped:
            print(f"[Controller] 夹爪已闭合，可能已抓住物体 (开合度: {state:.2f})")
        else:
            print(f"[Controller] 夹爪已闭合 (开合度: {state:.2f})")

        return True

    def go_home(self, speed: float = 0.5) -> bool:
        """回到初始位姿

        Args:
            speed: 速度因子

        Returns:
            是否成功
        """
        print("[Controller] 回到初始位姿...")
        return self.move_to_joints(self.HOME_QPOS, speed)

    def pick(self, target_pos: np.ndarray, approach_height: float = 0.15) -> bool:
        """执行抓取动作序列

        1. 打开夹爪
        2. 移动到目标上方（带姿态约束）
        3. 下降到抓取位置（带姿态约束）
        4. 闭合夹爪
        5. 提升到安全高度

        注: target_pos 为物体中心位置。IK 控制的是 attachment_site，
        而实际抓取接触点是 pad（在 attachment_site 下方 GRASP_OFFSET_Z）。
        因此需要将目标位置加上偏移来计算 attachment_site 的目标位置。

        Args:
            target_pos: 抓取目标位置 (3,) — 物体中心
            approach_height: 接近高度（相对于抓取位置 Z 的偏移量）

        Returns:
            是否成功抓取
        """
        target_pos = np.asarray(target_pos, dtype=float)

        # 将物体中心位置转换为 attachment_site 目标位置
        # pad 中心在 attachment_site 正下方 GRASP_OFFSET_Z
        # 所以 attachment_site 的 Z 要比物体中心高 GRASP_OFFSET_Z
        ee_target = target_pos.copy()
        ee_target[2] += self.GRASP_OFFSET_Z

        # 1. 先打开夹爪
        self.open_gripper()

        # 2. 移动到目标上方（带姿态约束确保朝下）
        approach_pos = ee_target.copy()
        approach_pos[2] += approach_height
        print(f"[Controller] 移动到接近位置: {approach_pos}")
        if not self.move_to_pose(approach_pos, self.GRASP_QUAT):
            return False

        # 3. 下降到抓取位置（带姿态约束，确保夹爪竖直朝下）
        print(f"[Controller] 下降到抓取位置: {ee_target} (pad 对准物体中心: {target_pos})")
        if not self.move_to_pose(ee_target, self.GRASP_QUAT):
            return False

        # 4. 闭合夹爪
        self.close_gripper(force=255)

        # 5. 提升
        print(f"[Controller] 提升到安全高度")
        if not self.move_to_pose(approach_pos, self.GRASP_QUAT):
            return False

        return True

    def place(self, target_pos: np.ndarray, approach_height: float = 0.15) -> bool:
        """执行放置动作序列

        1. 移动到目标上方（带姿态约束）
        2. 下降到目标位置（带姿态约束）
        3. 打开夹爪
        4. 提升到安全高度

        Args:
            target_pos: 放置目标位置 (3,) — 物体放置中心
            approach_height: 接近高度

        Returns:
            是否成功放置
        """
        target_pos = np.asarray(target_pos, dtype=float)

        # 将放置位置转换为 attachment_site 目标位置
        ee_target = target_pos.copy()
        ee_target[2] += self.GRASP_OFFSET_Z

        # 1. 移动到目标上方
        approach_pos = ee_target.copy()
        approach_pos[2] += approach_height
        print(f"[Controller] 移动到放置上方: {approach_pos}")
        if not self.move_to_pose(approach_pos, self.GRASP_QUAT):
            return False

        # 2. 下降
        print(f"[Controller] 下降到放置位置: {ee_target}")
        if not self.move_to_pose(ee_target, self.GRASP_QUAT):
            return False

        # 3. 打开夹爪释放
        self.open_gripper()

        # 4. 提升
        print(f"[Controller] 提升离开")
        if not self.move_to_pose(approach_pos, self.GRASP_QUAT):
            return False

        return True
