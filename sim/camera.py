"""
仿真相机模块
提供 RGB/Depth 图像渲染、视觉物体检测和 3D 坐标重建

支持两种物体检测方式:
1. 视觉检测 (Vision-based): 基于 RGB 颜色分割 + 深度相机 → 3D 世界坐标
2. 真值检测 (Ground-truth): 直接从仿真状态获取（用于对比验证）
"""

import mujoco
import numpy as np
import cv2
from pathlib import Path


class SimCamera:
    """仿真相机 - 用于场景感知

    支持 RGB 和深度图渲染，以及基于视觉的物体检测。
    """

    # 物体颜色配置 (HSV 范围)
    # 场景中的物体材质颜色:
    #   red_mat:    rgba="0.9 0.2 0.2 1"
    #   green_mat:  rgba="0.2 0.8 0.3 1"
    #   blue_mat:   rgba="0.2 0.3 0.9 1"
    #   yellow_mat: rgba="0.9 0.8 0.1 1"
    COLOR_CONFIGS = [
        {
            "name": "red_cube",
            "description": "红色方块",
            "ranges": [
                (np.array([0, 60, 60]), np.array([12, 255, 255])),
                (np.array([168, 60, 60]), np.array([180, 255, 255])),
            ],
        },
        {
            "name": "green_cube",
            "description": "绿色方块",
            "ranges": [
                # H=40~85 避免与黄色 (H≤34) 重叠; S>=40 接受较暗的绿色
                (np.array([40, 40, 40]), np.array([85, 255, 255])),
            ],
        },
        {
            "name": "blue_cylinder",
            "description": "蓝色圆柱",
            "ranges": [
                (np.array([95, 60, 60]), np.array([135, 255, 255])),
            ],
        },
        {
            "name": "yellow_cube",
            "description": "黄色方块",
            "ranges": [
                # H=15~34 避免与绿色 (H>=40) 重叠
                (np.array([15, 60, 60]), np.array([34, 255, 255])),
            ],
        },
    ]

    # 检测参数
    MIN_CONTOUR_AREA = 80  # 最小轮廓面积（像素）
    DEPTH_REGION_SIZE = 3  # 深度采样区域半径（像素）
    MAX_VALID_DEPTH = 3.0  # 最大有效深度（米）

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

    # ==================== 基础渲染 ====================

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
        """捕获深度图（原始深度缓冲值 0~1）

        Args:
            camera_name: 相机名称

        Returns:
            深度图 (H, W) float32，原始归一化值
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
            (rgb (H,W,3), depth_raw (H,W))
        """
        rgb = self.capture_rgb(camera_name)
        depth = self.capture_depth(camera_name)
        return rgb, depth

    # ==================== 相机几何模型 ====================

    def get_camera_intrinsics(self, camera_name: str) -> np.ndarray:
        """获取相机内参矩阵 K (3x3)

        根据 MuJoCo 的 fovy (垂直视场角) 和图像分辨率计算。
        MuJoCo 使用方形像素 (fx = fy)。

        Args:
            camera_name: 相机名称

        Returns:
            内参矩阵 K (3x3):
            [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
        """
        cam_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
        )
        fovy_deg = self.env.model.cam_fovy[cam_id]
        fovy_rad = np.deg2rad(fovy_deg)

        # 焦距（像素单位）
        fy = self.height / (2.0 * np.tan(fovy_rad / 2.0))
        fx = fy  # MuJoCo 方形像素
        cx = self.width / 2.0
        cy = self.height / 2.0

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ])
        return K

    def get_camera_extrinsics(self, camera_name: str) -> tuple[np.ndarray, np.ndarray]:
        """获取相机外参（在世界坐标系中的位姿）

        Args:
            camera_name: 相机名称

        Returns:
            (position (3,), rotation_matrix (3,3))
            rotation_matrix: 相机坐标系到世界坐标系的旋转矩阵
        """
        cam_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
        )
        pos = self.env.data.cam_xpos[cam_id].copy()
        rot = self.env.data.cam_xmat[cam_id].copy().reshape(3, 3)
        return pos, rot

    def depth_to_meters(self, depth_raw: np.ndarray) -> np.ndarray:
        """将 MuJoCo 深度渲染结果转换为真实深度（米）

        MuJoCo Python 渲染器 (mujoco.Renderer) 在 depth 模式下直接返回
        线性深度值（单位：米），表示从相机到场景表面的距离。
        无需进行 OpenGL 深度缓冲的非线性转换。

        仅将超远距离（背景/天空）的值设为 0 标记为无效。

        Args:
            depth_raw: 深度渲染结果 (H, W)，单位已经是米

        Returns:
            深度图 (H, W)，单位：米，无效区域为 0
        """
        depth_meters = depth_raw.copy()

        # 将超远距离设为 0（无效深度）
        # MuJoCo 远平面之外的区域深度值极大
        extent = self.env.model.stat.extent
        far = self.env.model.vis.map.zfar * extent
        depth_meters = np.where(depth_meters >= far * 0.99, 0.0, depth_meters)

        return depth_meters

    def pixel_to_world(
        self, u: float, v: float, depth: float, camera_name: str
    ) -> np.ndarray:
        """将像素坐标 + 深度转换为 3D 世界坐标

        使用 OpenGL 相机坐标系约定:
        - X 轴: 向右
        - Y 轴: 向上
        - Z 轴: 朝向观察者（远离场景）
        相机实际观看方向为 -Z。

        Args:
            u: 像素列坐标（水平）
            v: 像素行坐标（垂直，从顶部开始）
            depth: 该像素的真实深度（米）
            camera_name: 相机名称

        Returns:
            3D 世界坐标 (3,)
        """
        K = self.get_camera_intrinsics(camera_name)
        cam_pos, cam_rot = self.get_camera_extrinsics(camera_name)

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # 像素 → 相机坐标系
        # 图像 Y 轴向下，相机 Y 轴向上 → 需要翻转
        # 相机看向 -Z → z_cam = -depth
        x_cam = (u - cx) * depth / fx
        y_cam = -(v - cy) * depth / fy
        z_cam = -depth

        p_cam = np.array([x_cam, y_cam, z_cam])

        # 相机坐标系 → 世界坐标系
        p_world = cam_rot @ p_cam + cam_pos

        return p_world

    # ==================== 视觉物体检测 ====================

    def detect_objects_by_vision(
        self, camera_name: str = "wrist_cam"
    ) -> tuple[list[dict], np.ndarray, np.ndarray]:
        """使用视觉方法检测物体（颜色分割 + 深度重建）

        模拟真实世界的视觉感知流水线：
        1. 从腕部相机捕获 RGB + 深度图
        2. HSV 颜色空间分割
        3. 轮廓检测定位物体像素区域
        4. 深度查找 + 相机投影 → 3D 世界坐标

        Args:
            camera_name: 使用的相机名称

        Returns:
            (detected_objects, rgb_image, depth_meters)
            - detected_objects: 检测到的物体列表
            - rgb_image: RGB 图像 (H,W,3)
            - depth_meters: 线性化深度图 (H,W)
        """
        # 确保 mj_forward 已调用（相机位姿需要最新状态）
        mujoco.mj_forward(self.env.model, self.env.data)

        # 1. 捕获 RGBD
        rgb, depth_raw = self.capture_rgbd(camera_name)
        depth_meters = self.depth_to_meters(depth_raw)

        # 2. RGB → HSV 用于颜色分割
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        # 3. 对每种颜色物体进行检测
        detected = []
        for config in self.COLOR_CONFIGS:
            result = self._detect_single_object(
                config, hsv, depth_meters, camera_name
            )
            if result is not None:
                detected.append(result)

        if detected:
            print(f"[Camera] 视觉检测完成: 检测到 {len(detected)} 个物体")
            for obj in detected:
                pos = obj["position"]
                print(
                    f"  - {obj['description']} ({obj['name']}): "
                    f"位置 ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}), "
                    f"深度 {obj['depth']:.4f}m"
                )
        else:
            print("[Camera] 视觉检测完成: 未检测到物体")

        return detected, rgb, depth_meters

    def _detect_single_object(
        self,
        color_config: dict,
        hsv: np.ndarray,
        depth_meters: np.ndarray,
        camera_name: str,
    ) -> dict | None:
        """检测单个颜色的物体

        Args:
            color_config: 颜色配置 {"name", "description", "ranges"}
            hsv: HSV 图像
            depth_meters: 深度图（米）
            camera_name: 相机名称

        Returns:
            检测结果字典，未检测到返回 None
        """
        # 合并所有颜色范围的掩码（如红色需要两个范围）
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in color_config["ranges"]:
            mask |= cv2.inRange(hsv, lower, upper)

        # 形态学操作去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # 取最大轮廓
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < self.MIN_CONTOUR_AREA:
            return None

        # 计算质心
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None

        cu = int(M["m10"] / M["m00"])
        cv_coord = int(M["m01"] / M["m00"])

        # 在质心附近区域取深度中值（抗噪）
        r = self.DEPTH_REGION_SIZE
        v_min = max(0, cv_coord - r)
        v_max = min(self.height, cv_coord + r + 1)
        u_min = max(0, cu - r)
        u_max = min(self.width, cu + r + 1)

        depth_region = depth_meters[v_min:v_max, u_min:u_max]
        valid_depths = depth_region[(depth_region > 0) & (depth_region < self.MAX_VALID_DEPTH)]

        if len(valid_depths) == 0:
            return None

        d = float(np.median(valid_depths))

        # 像素 + 深度 → 3D 世界坐标
        world_pos = self.pixel_to_world(cu, cv_coord, d, camera_name)

        # 计算边界框
        bx, by, bw, bh = cv2.boundingRect(largest)

        # 简单置信度（基于面积）
        confidence = min(1.0, area / 500.0)

        return {
            "name": color_config["name"],
            "description": color_config["description"],
            "position": [
                round(float(world_pos[0]), 4),
                round(float(world_pos[1]), 4),
                round(float(world_pos[2]), 4),
            ],
            "depth": round(d, 4),
            "pixel_center": [cu, cv_coord],
            "pixel_area": int(area),
            "bbox": [bx, by, bw, bh],
            "confidence": round(confidence, 2),
        }

    # ==================== 检测可视化 ====================

    def save_detection_image(
        self,
        rgb: np.ndarray,
        detected_objects: list[dict],
        filepath: str,
    ):
        """将检测结果标注到 RGB 图像并保存

        在图像上绘制每个检测到的物体的边界框、名称和 3D 坐标。

        Args:
            rgb: RGB 图像 (H,W,3)
            detected_objects: detect_objects_by_vision 返回的物体列表
            filepath: 保存路径
        """
        # 转换为 BGR（OpenCV 格式）并拷贝
        vis = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)

        # 颜色映射
        color_map = {
            "red_cube": (0, 0, 255),       # BGR: 红
            "green_cube": (0, 200, 0),      # BGR: 绿
            "blue_cylinder": (255, 100, 0), # BGR: 蓝
            "yellow_cube": (0, 200, 255),   # BGR: 黄
        }

        for obj in detected_objects:
            name = obj["name"]
            pos = obj["position"]
            bx, by, bw, bh = obj["bbox"]
            color = color_map.get(name, (255, 255, 255))

            # 绘制边界框
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), color, 2)

            # 绘制质心
            cx, cy = obj["pixel_center"]
            cv2.circle(vis, (cx, cy), 5, color, -1)

            # 标注文字（使用英文名称，OpenCV 默认不支持中文渲染）
            label = name
            coord_text = f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"

            cv2.putText(
                vis, label,
                (bx, by - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )
            cv2.putText(
                vis, coord_text,
                (bx, by - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
            )

        cv2.imwrite(filepath, vis)
        print(f"[Camera] 检测结果已保存: {filepath}")

    # ==================== 场景描述生成 ====================

    def get_vision_scene_description(
        self, detected_objects: list[dict], camera_name: str = "wrist_cam"
    ) -> str:
        """根据视觉检测结果生成场景文本描述，供 LLM Agent 使用

        Args:
            detected_objects: 视觉检测到的物体列表
            camera_name: 使用的相机名称

        Returns:
            场景的自然语言描述
        """
        # 获取机械臂状态
        ee_pos, _ = self.env.get_ee_pose()
        gripper_state = self.env.get_gripper_state()
        gripper_status = "闭合" if gripper_state > 0.5 else "张开"

        # 获取相机位姿
        cam_pos, _ = self.get_camera_extrinsics(camera_name)

        lines = []
        lines.append("=== 场景信息（腕部相机视觉检测）===")
        lines.append(f"\n** 感知方式: 腕部相机 RGB-D 视觉检测 **")
        lines.append(
            f"相机位置: ({cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f})"
        )
        lines.append(
            f"机械臂末端位置: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})"
        )
        lines.append(f"夹爪状态: {gripper_status} (开合度: {gripper_state:.2f})")
        lines.append(f"\n桌面高度: 0.37m")
        lines.append(f"桌面范围: X[0.15, 0.85], Y[-0.50, 0.50]")

        if detected_objects:
            lines.append(f"\n检测到 {len(detected_objects)} 个物体:")
            for obj in detected_objects:
                pos = obj["position"]
                conf = obj.get("confidence", 0)
                lines.append(
                    f"  - {obj['description']} ({obj['name']}): "
                    f"位置 ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
                    f"深度 {obj['depth']:.3f}m, "
                    f"置信度 {conf:.0%}"
                )
        else:
            lines.append("\n未检测到任何物体。可能原因:")
            lines.append("  - 相机视野外")
            lines.append("  - 物体被遮挡")
            lines.append("  - 需要移动到更好的扫描位置")

        lines.append(
            f"\n坐标系说明: X正方向=远离机器人基座, Y正方向=左侧, Z正方向=向上"
        )
        lines.append(f"安全抓取高度: Z=0.50m (接近时先到此高度)")
        lines.append(f"物体放置高度: Z≈0.39m (桌面 + 物体半高)")

        return "\n".join(lines)

    # ==================== 真值检测（向后兼容）====================

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
                "half_height": info.get("half_height", 0.0),
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
            hh = obj["half_height"]
            top_z = pos[2] + hh
            status = "在桌面上" if obj["on_table"] else "不在桌面上"
            lines.append(
                f"  - {obj['description']} ({obj['name']}): "
                f"位置中心 ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
                f"物体半高 {hh:.3f}m, 顶部高度Z={top_z:.3f}m, {status}"
            )

        lines.append(f"\n坐标系说明: X正方向=远离机器人基座, Y正方向=左侧, Z正方向=向上")
        lines.append(f"安全抓取高度: Z=0.50m (接近时先到此高度)")
        lines.append(f"抓取高度: Z=0.39m (与桌面齐平)")

        return "\n".join(lines)

    # ==================== 工具方法 ====================

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
