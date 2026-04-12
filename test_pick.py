"""最终测试: 旋转后的基座 + 原始 home IK"""
import numpy as np
from sim.environment import SimEnvironment
from sim.controller import RobotController

env = SimEnvironment(model_path="models/scene.xml", render=False)
ctrl = RobotController(env)

home = ctrl.HOME_QPOS.copy()
quat = np.array([0.0, 1.0, 0.0, 0.0])

print(f"Home: {np.degrees(home)}")
ee_pos, _ = env.get_ee_pose()
print(f"Home EE pos: {ee_pos}")

print("\n=== IK 测试 (从 home 开始) ===")
for name in ["red_cube", "green_cube", "blue_cylinder", "yellow_cube"]:
    pos, _ = env.get_body_pose(name)
    ee_target = pos.copy()
    ee_target[2] += ctrl.GRASP_OFFSET_Z
    
    try:
        q = ctrl.inverse_kinematics(ee_target, quat, init_qpos=home)
        diff = np.degrees(np.max(np.abs(q - home)))
        print(f"  {name}: ✅ 最大关节变化={diff:.1f}°")
    except Exception as e:
        print(f"  {name}: ❌ {e}")

print("\n=== 完整 pick+place 测试 (green_cube) ===")
env2 = SimEnvironment(model_path="models/scene.xml", render=False)
ctrl2 = RobotController(env2)

green_pos, _ = env2.get_body_pose("green_cube")
print(f"green_cube pos: {green_pos}")

success = ctrl2.pick(green_pos)
print(f"pick: {'✅' if success else '❌'}")

green_after, _ = env2.get_body_pose("green_cube")
lifted = green_after[2] > green_pos[2] + 0.02
print(f"物体提起: {lifted} (Z: {green_pos[2]:.3f} -> {green_after[2]:.3f})")
print(f"夹爪: {env2.get_gripper_state():.2f}")

env.close()
env2.close()
