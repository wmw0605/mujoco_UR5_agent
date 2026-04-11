# UR5e + Qwen Agent 自主抓取控制系统

## 简介
本项目是一个整合了 UR5e 机械臂、Robotiq 2F-85 夹爪和 Qwen 大语言模型的自主抓取系统。它可以接收自然语言指令，通过 Function Calling 机制调用工具，自主规划动作序列，并在 MuJoCo 仿真环境或真实机器人上执行抓取和放置等操作。

## 核心特性
- **双工作模式支持**: 支持 MuJoCo 仿真环境（默认）与真实 UR5e 机械臂环境的切换，一套代码能够无缝迁移。
- **大模型决策 (Qwen Agent)**: 接入阿里云 DashScope 兼容的 OpenAI 接口，默认使用 Qwen-max，通过自然语言对话智能调度底层机器人的规划与动作。
- **丰富的机器人技能**: 将繁杂的底层操作高度封装，内置 `get_scene_info`, `get_robot_state`, `move_to`, `grasp`, `release`, `go_home`, `pick_and_place` 等多种基础与高级工具。
- **CLI 交互式控制台**: 提供一个便捷的人机交互终端，既可以直接输入指令与 Agent 对话，也支持 `home`, `scene`, `state`、`reset` 等内置快捷调试命令。

## 系统要求及安装

### 环境依赖
项目基于 `Python >= 3.10` 构建。主要依赖包涵盖：
- `mujoco >= 3.6.0`
- `openai >= 2.31.0`
- `opencv-python >= 4.13.0.92`
- `scipy >= 1.15.3`

### 安装步骤
推荐使用 [uv](https://github.com/astral-sh/uv) （或者 pip）进行依赖安装：
```bash
# 激活环境后使用 uv 同步依赖
uv sync
```
或者使用 pip：
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 配置说明

在使用基于大语言模型的 Agent 智能体功能前，需先配置阿里云 DashScope API 密钥。可以通过环境变量配置：
```bash
export DASHSCOPE_API_KEY="your-api-key"
```
（或者直接在 `config.py` 中或实例化 `GraspingAgent` 时传入设定）。

## 运行与使用

你可以直接运行 `main.py` 进入控制系统：

### 1. 默认仿真模式
```bash
# 包含启动仿真环境、加载场景和开启 GUI 渲染，配置默认 qwen-max 模型
python main.py
```

### 2. 指定不同的大模型
```bash
# 可选模型包括 qwen-max, qwen-plus, qwen-turbo 等
python main.py --model qwen-plus
```

### 3. 连接真实机器人
```bash
# 以真实机械臂模式启动
python main.py --mode real --robot-ip 192.168.1.100
```

### 4. 无头模式 (关闭仿真 GUI 渲染)
```bash
python main.py --no-render
```

### 5. 单次指令执行 (非交互模式)
```bash
python main.py --task "抓取红色方块并将其放置到蓝色盒子里"
```

## 在交互终端中的特殊命令
当在 CLI 控制台中出现 `🤖 请输入任务指令 >` 提示符时，你可以用自然语言与 Agent 对话，或者直接输入以下特殊指令：
- `home`：让机械臂复位到初始位姿
- `scene`：打印获取当前场景中的所有检测物体及其位姿信息
- `state`：打印当前机器人的工作状态及各个轴的关节信息
- `reset`：重置 Agent 对话历史，以清理多层对话带来的记忆包袱
- `quit` 或 `exit`：平稳关闭模拟器及退出整个程序

## 架构说明
* `main.py`: 项目主入口，负责配置初始化、加载所选配置模式以及启动核心终端交互循环。
* `config.py`: 主要结构化配置数据。
* `agent/`: Qwen Agent 会话与工作流。
  * `agent.py`: 整合 OpenAI 兼容接口，执行多轮对话与工具解析反馈，控制工作深度限制。
  * `prompts.py`: 控制 Agent 工作流设定的主要 System Persona 与先验知识文本。
  * `tools.py`: 各具体机器人功能的工具封装与对接方法。
* `robot/`: 底层机器人的接口统一封装层。
  * `base.py`: 基类及接口核心协议。
  * `sim_robot.py`: 面向 MuJoCo 仿真的硬件模拟接口实现。
  * `real_robot.py`: 面向实际 UR5e 环境下具体物理服务端的接口实现。
* `sim/`: 负责驱动 Mujoco 渲染器、建立环境约束与动力学基础。
  * `environment.py`, `camera.py`, `controller.py`: 环境控制层与伺服算法等内容。
* `models/`: 存放所需的静态物理资产如 `.xml` 与其余物体的贴图、网格定义等。
