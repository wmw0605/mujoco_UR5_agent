"""
腕部相机视觉检测测试脚本

测试流程:
1. 初始化仿真环境
2. 通过 SimRobot 的 scan_workspace 执行多位置扫描
3. 将检测结果与仿真真值坐标对比
4. 保存可视化结果

运行: python test_wrist_camera.py
"""

import numpy as np
from robot.sim_robot import SimRobot


def main():
    print("=" * 60)
    print("  腕部相机视觉检测系统测试")
    print("=" * 60)

    # 1. 初始化仿真机器人（含环境、控制器、相机）
    print("\n[1] 初始化仿真机器人...")
    robot = SimRobot(model_path="models/scene.xml", render=True)

    # 2. 执行多位置扫描
    print("\n[2] 执行腕部相机扫描...")
    result = robot.scan_workspace()

    # 3. 输出扫描结果
    print(f"\n[3] 扫描状态: {'成功' if result['success'] else '失败'}")
    print(f"    消息: {result['message']}")

    # 4. 与真值对比
    print("\n[4] 视觉检测 vs 仿真真值：")
    gt_objects = robot.camera.detect_objects()
    detected = result["objects"]

    print(f"{'物体':<16} {'真值 (x, y, z)':<32} {'检测 (x, y, z)':<32} {'误差(mm)'}")
    print("-" * 95)

    errors = []
    for gt_obj in gt_objects:
        gt_pos = gt_obj["position"]
        det = None
        for d in detected:
            if d["name"] == gt_obj["name"]:
                det = d
                break

        if det:
            det_pos = det["position"]
            error = np.linalg.norm(np.array(gt_pos) - np.array(det_pos)) * 1000
            errors.append(error)
            print(
                f"{gt_obj['name']:<16} "
                f"({gt_pos[0]:.4f}, {gt_pos[1]:.4f}, {gt_pos[2]:.4f})    "
                f"({det_pos[0]:.4f}, {det_pos[1]:.4f}, {det_pos[2]:.4f})    "
                f"{error:.1f}"
            )
        else:
            print(
                f"{gt_obj['name']:<16} "
                f"({gt_pos[0]:.4f}, {gt_pos[1]:.4f}, {gt_pos[2]:.4f})    "
                f"{'未检测到':<32} {'N/A'}"
            )

    if errors:
        print(f"\n    平均误差: {np.mean(errors):.1f}mm")
        print(f"    最大误差: {np.max(errors):.1f}mm")

    # 5. 输出完整场景描述
    print("\n[5] 场景描述:")
    print(result["description"])

    # 6. 回到初始位姿
    print("\n[6] 回到初始位姿...")
    robot.controller.go_home()

    # 7. 等待用户关闭窗口
    print("\n测试完成! 关闭仿真窗口退出。")
    print("    检测结果图: detection_result.png")

    import time
    while robot.is_running():
        robot.env.step(10)
        time.sleep(0.05)

    robot.close()


if __name__ == "__main__":
    main()
