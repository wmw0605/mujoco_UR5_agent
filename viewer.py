"""
MuJoCo GUI 可视化启动脚本

用法:
  .venv/bin/python viewer.py                 # 使用 MuJoCo 自带 Viewer
  .venv/bin/python viewer.py --passive        # 使用被动模式 Viewer（可编程控制）

注意: 需要有图形桌面环境（X11/Wayland）才能显示 GUI。
"""

import argparse
import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description="UR5e 仿真可视化")
    parser.add_argument("--passive", action="store_true", help="使用被动模式 Viewer")
    parser.add_argument("--model", default="models/scene.xml", help="模型文件路径")
    args = parser.parse_args()

    # 加载模型
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = Path(__file__).parent / model_path

    print(f"加载模型: {model_path}")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # 重置到 home 关键帧
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    print("模型加载成功!")
    print(f"  自由度: {model.nv}")
    print(f"  执行器: {model.nu}")
    print(f"  物体数: {model.nbody}")

    if args.passive:
        # 被动模式：可以同时在代码中控制仿真
        print("\n启动被动模式 Viewer...")
        print("(可以在代码中操控机器人，关闭窗口退出)")

        viewer = mujoco.viewer.launch_passive(
            model, data, show_left_ui=True, show_right_ui=True
        )

        # 简单的控制循环示例
        print("\n机械臂将缓慢移动到桌面上方...")
        target_ctrl = np.array([-1.5708, -1.2, 1.8, -1.5708, -1.5708, 0.0])

        step = 0
        while viewer.is_running():
            # 平滑插值到目标
            alpha = min(1.0, step * 0.0001)
            home_ctrl = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
            ctrl = home_ctrl + alpha * (target_ctrl - home_ctrl)

            for i in range(6):
                data.ctrl[i] = ctrl[i]

            mujoco.mj_step(model, data)
            viewer.sync()
            step += 1

        print("Viewer 已关闭")

    else:
        # 主动模式：直接启动 MuJoCo 自带的交互式 Viewer
        print("\n启动 MuJoCo 交互式 Viewer...")
        print("操作说明:")
        print("  鼠标左键: 旋转视角")
        print("  鼠标右键: 平移视角")
        print("  滚轮: 缩放")
        print("  双击物体: 选中")
        print("  Ctrl+右键: 施加力")
        print("  Space: 暂停/继续")
        print("  Backspace: 重置")
        print("  ESC: 退出")

        mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
