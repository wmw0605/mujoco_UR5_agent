"""
真实机器人实现（框架）
基于 ur_rtde 控制 UR5 + Robotiq 2F-85

注意: 此模块需要额外安装依赖:
  uv add ur-rtde pymodbus

仅在连接真实机器人时使用。
"""

import numpy as np
from robot.base import RobotInterface


class RealRobot(RobotInterface):
    """真实机器人实现

    通过 ur_rtde 控制 UR5，通过 Modbus/URScript 控制 Robotiq 2F-85。

    TODO: 根据实际硬件配置完善此类
    """

    def __init__(
        self,
        robot_ip: str = "192.168.1.100",
        gripper_port: str = "/dev/ttyUSB0",
    ):
        """初始化真实机器人连接

        Args:
            robot_ip: UR5 的 IP 地址
            gripper_port: Robotiq 夹爪的串口端口
        """
        self.robot_ip = robot_ip
        self.gripper_port = gripper_port

        try:
            import rtde_control
            import rtde_receive
        except ImportError:
            raise ImportError(
                "需要安装 ur_rtde 库: uv add ur-rtde\n"
                "请确保已正确安装 ur_rtde 及其依赖。"
            )

        print(f"[RealRobot] 连接 UR5: {robot_ip}")
        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)

        # TODO: 初始化 Robotiq 夹爪
        # self._init_gripper()

        # TODO: 初始化真实相机
        # self._init_camera()

        print("[RealRobot] 真实机器人连接完成")

    def get_tcp_pose(self) -> dict:
        """获取 TCP 位姿"""
        pose = self.rtde_r.getActualTCPPose()
        return {
            "position": list(pose[:3]),
            "orientation": list(pose[3:]),
        }

    def get_joint_positions(self) -> list[float]:
        """获取关节角度"""
        return list(self.rtde_r.getActualQ())

    def get_gripper_state(self) -> dict:
        """获取夹爪状态"""
        # TODO: 通过 Modbus 读取 Robotiq 状态
        return {
            "is_open": True,
            "openness": 1.0,
        }

    def move_to_position(
        self,
        x: float, y: float, z: float,
        rx: float = None, ry: float = None, rz: float = None,
        speed: float = 0.5,
    ) -> dict:
        """移动到目标位置"""
        # 获取当前姿态作为默认值
        current = self.rtde_r.getActualTCPPose()
        if rx is None:
            rx = current[3]
        if ry is None:
            ry = current[4]
        if rz is None:
            rz = current[5]

        target = [x, y, z, rx, ry, rz]
        velocity = speed * 0.5  # m/s
        acceleration = speed * 0.5  # m/s^2

        try:
            self.rtde_c.moveL(target, velocity, acceleration)
            final = self.rtde_r.getActualTCPPose()
            return {
                "success": True,
                "message": f"已移动到 ({final[0]:.3f}, {final[1]:.3f}, {final[2]:.3f})",
                "final_position": list(final[:3]),
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"移动失败: {e}",
                "final_position": list(current[:3]),
            }

    def open_gripper(self) -> dict:
        """打开夹爪"""
        # TODO: 通过 URScript 或 Modbus 控制 Robotiq
        # 示例 URScript:
        # self.rtde_c.sendCustomScript("rq_open()")
        return {
            "success": True,
            "message": "夹爪已打开（TODO: 实现真实控制）",
        }

    def close_gripper(self) -> dict:
        """闭合夹爪"""
        # TODO: 实现真实夹爪控制
        return {
            "success": True,
            "message": "夹爪已闭合（TODO: 实现真实控制）",
            "object_grasped": False,
        }

    def go_home(self) -> dict:
        """回到初始位姿"""
        home_joints = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
        try:
            self.rtde_c.moveJ(home_joints, 0.5, 0.5)
            return {
                "success": True,
                "message": "已回到初始位姿",
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"回到初始位姿失败: {e}",
            }

    def get_scene_info(self) -> str:
        """获取场景信息"""
        # TODO: 使用真实相机 + 视觉检测
        tcp = self.get_tcp_pose()
        pos = tcp["position"]
        return (
            "=== 当前场景信息 ===\n"
            f"机械臂末端位置: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})\n"
            "注意: 真实场景物体检测尚未实现，请使用视觉系统获取物体信息。"
        )

    def pick_object(self, object_name: str) -> dict:
        """抓取物体"""
        # TODO: 配合视觉系统定位物体
        return {
            "success": False,
            "message": f"真实抓取 {object_name} 尚未实现，需要视觉系统配合",
        }

    def place_object(self, x: float, y: float, z: float) -> dict:
        """放置物体"""
        # 先移动到目标位置上方
        result = self.move_to_position(x, y, z + 0.1)
        if not result["success"]:
            return result

        # 下降
        result = self.move_to_position(x, y, z)
        if not result["success"]:
            return result

        # 打开夹爪
        self.open_gripper()

        # 提升
        self.move_to_position(x, y, z + 0.1)

        return {
            "success": True,
            "message": f"已放置到 ({x:.3f}, {y:.3f}, {z:.3f})",
        }

    def close(self):
        """断开连接"""
        try:
            self.rtde_c.stopScript()
            print("[RealRobot] 已断开连接")
        except Exception:
            pass
