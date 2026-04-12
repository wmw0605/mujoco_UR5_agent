"""调试: 检查抓取时的精确位置"""
import numpy as np
import mujoco
from sim.environment import SimEnvironment
from sim.controller import RobotController

env = SimEnvironment(model_path="models/scene.xml", render=False)
ctrl = RobotController(env)

green_pos, _ = env.get_body_pose("green_cube")
print(f"green_cube 位置: {green_pos}")

# 打开夹爪
ctrl.open_gripper()

# 到抓取位置
ee_target = green_pos.copy()
ee_target[2] += ctrl.GRASP_OFFSET_Z
print(f"EE 目标: {ee_target}")

# approach
approach = ee_target.copy()
approach[2] += 0.03
ctrl.move_to_pose(approach, ctrl.GRASP_QUAT)

# 到抓取位置
ctrl.move_to_pose(ee_target, ctrl.GRASP_QUAT)

# 检查实际位置
ee_pos, _ = env.get_ee_pose()
pinch_pos, _ = env.get_pinch_pose()
green_now, _ = env.get_body_pose("green_cube")

print(f"\n=== 到达抓取位置后 ===")
print(f"EE 实际: {ee_pos}")
print(f"EE 目标: {ee_target}")
print(f"EE 偏差: {ee_pos - ee_target}")
print(f"pinch 实际: {pinch_pos}")
print(f"green 实际: {green_now}")

# pad 位置
for pad_name in ["right_pad", "left_pad"]:
    bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, pad_name)
    print(f"{pad_name}: {env.data.xpos[bid]}")

# 检查EE Z轴
rot_mat = env.data.site_xmat[env._ee_site_id].reshape(3,3)
print(f"EE Z轴: {rot_mat[:, 2]}")

# gripper base
gb_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
print(f"gripper_base: {env.data.xpos[gb_id]}")

# pad 高度 vs 物体高度分析
for pad_name in ["right_pad", "left_pad"]:
    bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, pad_name)
    pad_pos = env.data.xpos[bid]
    print(f"\n{pad_name} 与 green_cube 对比:")
    print(f"  pad XY: ({pad_pos[0]:.4f}, {pad_pos[1]:.4f})")
    print(f"  obj XY: ({green_now[0]:.4f}, {green_now[1]:.4f})")
    print(f"  pad Z: {pad_pos[2]:.4f}, obj Z: {green_now[2]:.4f}, 差: {pad_pos[2]-green_now[2]:.4f}")

env.close()
