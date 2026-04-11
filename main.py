"""
UR5e + Qwen Agent 自主抓取仿真系统
主入口文件

用法:
  # 仿真模式（默认）
  python main.py

  # 仿真模式，指定模型
  python main.py --mode sim --model qwen-plus

  # 真实机器人模式
  python main.py --mode real --robot-ip 192.168.1.100

  # 无渲染模式（无头）
  python main.py --no-render

环境变量:
  DASHSCOPE_API_KEY: 阿里云 DashScope API Key
"""

import argparse
import sys
import signal

from config import Config


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="UR5e + Qwen Agent 自主抓取仿真系统"
    )
    parser.add_argument(
        "--mode",
        choices=["sim", "real"],
        default="sim",
        help="运行模式: sim=仿真, real=真实机器人 (默认: sim)",
    )
    parser.add_argument(
        "--model",
        default="qwen-max",
        help="Qwen 模型名称 (默认: qwen-max)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="禁用 GUI 渲染",
    )
    parser.add_argument(
        "--robot-ip",
        default="192.168.1.100",
        help="真实机器人 IP 地址 (默认: 192.168.1.100)",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="直接执行指定任务（非交互模式）",
    )
    return parser.parse_args()


def create_robot(config: Config):
    """根据配置创建机器人实例"""
    if config.mode == "sim":
        from robot.sim_robot import SimRobot
        return SimRobot(
            model_path=config.model_path,
            render=config.render,
        )
    elif config.mode == "real":
        from robot.real_robot import RealRobot
        return RealRobot(
            robot_ip=config.robot_ip,
            gripper_port=config.gripper_port,
        )
    else:
        raise ValueError(f"未知运行模式: {config.mode}")


def interactive_loop(agent, robot):
    """交互式命令循环"""
    print("\n" + "=" * 60)
    print("  UR5e + Qwen Agent 自主抓取系统")
    print("  输入任务指令让 Agent 自动执行")
    print("  特殊命令:")
    print("    'quit' / 'exit'  - 退出系统")
    print("    'reset'          - 重置 Agent 对话历史")
    print("    'home'           - 机械臂回到初始位姿")
    print("    'scene'          - 查看当前场景信息")
    print("    'state'          - 查看机器人状态")
    print("=" * 60 + "\n")

    while True:
        try:
            # 检查仿真是否仍在运行
            if hasattr(robot, "is_running") and not robot.is_running():
                print("\n[系统] 仿真窗口已关闭，退出程序")
                break

            user_input = input("\n🤖 请输入任务指令 > ").strip()

            if not user_input:
                continue

            # 特殊命令
            if user_input.lower() in ("quit", "exit", "q"):
                print("\n[系统] 正在退出...")
                break

            elif user_input.lower() == "reset":
                agent.reset()
                print("[系统] Agent 对话历史已重置")
                continue

            elif user_input.lower() == "home":
                result = robot.go_home()
                print(f"[系统] {result['message']}")
                continue

            elif user_input.lower() == "scene":
                info = robot.get_scene_info()
                print(info)
                continue

            elif user_input.lower() == "state":
                state = robot.get_robot_state()
                import json
                print(json.dumps(state, ensure_ascii=False, indent=2))
                continue

            # 发送任务给 Agent
            response = agent.run(user_input)

        except KeyboardInterrupt:
            print("\n\n[系统] 收到中断信号，正在退出...")
            break
        except Exception as e:
            print(f"\n[系统] 错误: {e}")
            import traceback
            traceback.print_exc()
            continue


def main():
    """主函数"""
    args = parse_args()
    config = Config.from_args(args)

    print("=" * 60)
    print(f"  运行模式: {'仿真' if config.mode == 'sim' else '真实机器人'}")
    print(f"  Qwen 模型: {config.qwen_model}")
    print(f"  GUI 渲染: {'开启' if config.render else '关闭'}")
    if config.mode == "real":
        print(f"  机器人 IP: {config.robot_ip}")
    print("=" * 60)

    # 创建机器人
    print("\n[系统] 初始化机器人...")
    robot = create_robot(config)

    # 创建 Agent
    print("[系统] 初始化 Qwen Agent...")
    try:
        from agent.agent import GraspingAgent
        agent = GraspingAgent(
            robot=robot,
            model=config.qwen_model,
            api_key=config.dashscope_api_key or None,
            base_url=config.dashscope_base_url,
        )
    except ValueError as e:
        print(f"\n[警告] Agent 初始化失败: {e}")
        print("[系统] 将以无 Agent 模式运行（仅支持基础命令）")
        agent = None

    # 注册退出信号
    def signal_handler(sig, frame):
        print("\n[系统] 正在关闭...")
        robot.close()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    if args.task and agent:
        # 非交互模式：直接执行任务
        response = agent.run(args.task)
        print(f"\n[最终结果] {response}")
    else:
        if agent is None:
            # 没有 Agent，提供基础交互
            print("\n[系统] 无 Agent 模式。可用命令: scene, state, home, quit")
            class DummyAgent:
                def run(self, instruction):
                    return "Agent 未初始化，请设置 DASHSCOPE_API_KEY 环境变量"
                def reset(self):
                    pass
            agent = DummyAgent()

        # 交互模式
        interactive_loop(agent, robot)

    # 清理
    robot.close()
    print("[系统] 系统已关闭")


if __name__ == "__main__":
    main()
