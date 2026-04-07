[![中文](https://img.shields.io/badge/中文-简体-blue)](./README_CN.md)  
[![English](https://img.shields.io/badge/English-English-green)](./README.md)

# embodia

`embodia` 是一个面向机器人和模型的轻量 Python 运行时接口库。它只解决一个核心问题：让不同项目里的 robot 类和 policy 类说同一种数据流语言。它不做训练框架，不做服务系统，不依赖 ROS，也不试图变成大而全平台。

对大多数用户来说，真正需要接触的表面应该很小：

1. 继承 `RobotMixin` / `PolicyMixin`
2. 用 `from_yaml(...)` 做接口对齐
3. 用 `run_step(...)` 跑统一数据流
4. 需要推理期能力时再加 `InferenceRuntime(...)`

其他能力存在，但都只是为了支撑这条主路径。

## 安装

```bash
git clone https://github.com/zywang03/embodia.git
cd embodia
pip install .
```

如果你想使用 `from_yaml(...)`：

```bash
pip install ".[yaml]"
```

如果你想使用可选的 OpenPI 远程策略能力：

```bash
pip install ".[openpi-remote]"
```

`embodia` 本身不要求 `numpy` 或 `torch`。如果用户项目已经安装了这些库，embodia 可以在运行时边界接受它们并归一化成自己的核心数据结构。

## 快速开始

把 embodia 放在你现有类的最外层，原来的 native 方法保持不变：

```python
import embodia as em


class YourRobot(em.RobotMixin):
    def capture(self): ...
    def send_command(self, action): ...
    def home(self): ...


class YourPolicy(em.PolicyMixin):
    def clear_state(self): ...
    def infer(self, frame): ...
```

然后从 YAML 加载接口对齐配置：

```python
robot = YourRobot.from_yaml("docs/yaml_config_example.yml")
policy = YourPolicy.from_yaml("docs/yaml_config_example.yml")
```

这个 YAML 只描述共享 schema 和方法别名，不承载构造参数。对 policy 来说，
embodia 会直接从 `schema:` 推导输入输出。如果模型需要额外条件输入，比如
prompt，放到 `Frame.task` 里。robot spec 不再声明 task 相关能力。

### YAML 和你的方法到底是什么关系

`schema:` 定义的是标准运行时字段名，`method_aliases:` 只负责告诉 embodia
你现有类里的哪个方法对应 `observe`、`act`、`reset`、`infer`。它不会帮你生成
实现，也不会改你的构造函数。

如果你没有额外声明 Python 侧的 `MODALITY_MAPS`，那你的 native 方法就应该直接
使用 YAML 里写的那些名字。也就是说，YAML 本身就是运行时契约，embodia 会按它来
校验输入输出。

对应关系是：

- `schema.images` -> `frame.images` 里应该出现的键
- `schema.components.<name>.state` -> `frame.state` 里应该出现的键
- `schema.components.<name>` -> `Command.target`
- `schema.components.<name>.command_kinds` -> `Command.kind`
- `schema.task` -> 仅 policy 侧使用的 `frame.task` 键

所以最常见的方法契约就是：

- `capture()` / `observe()` 返回 `em.Frame` 或 frame-like `dict`，里面要有
  `timestamp_ns`、`images`、`state`。`images` 的键来自 `schema.images`，
  `state` 的键来自所有 `schema.components[*].state` 的并集。
- robot 的 `home()` / `reset()` 返回结构和 `capture()` 一样。
- `send_command(action)` / `act(action)` 接收的是一个 `em.Action`。其中
  `action.commands[*].target` 就是 YAML 里的组件名，
  `action.commands[*].kind` 就是这些组件声明的命令类型。
- policy 的 `clear_state()` / `reset()` 必须返回 `None`。
- `infer(frame)` 接收的是一个 `em.Frame`。`frame.images` 和 `frame.state`
  已经按 YAML 对齐好了；如果声明了 `schema.task`，额外条件会出现在
  `frame.task` 里。`infer(frame)` 应该返回 `em.Action` 或 action-like `dict`，
  其 commands 也必须对齐到 YAML 里的组件定义。

对当前仓库里的示例 YAML 来说，运行时结构应该是：

```python
# robot.capture() / robot.home() 返回这种结构
{
    "timestamp_ns": 1710000000000000000,
    "images": {
        "front_rgb": ...,
    },
    "state": {
        "joint_positions": [...],
        "position": 0.5,
    },
}

# policy.infer(frame) 返回这种结构
{
    "commands": [
        {
            "target": "arm",
            "kind": "cartesian_pose_delta",
            "value": [...],
        },
        {
            "target": "gripper",
            "kind": "gripper_position",
            "value": [0.5],
        },
    ],
    "dt": 0.1,
}
```

如果你的原项目里名字不是这些标准名，那就把映射保留在 Python 代码里，用
`MODALITY_MAPS` 处理。embodia 会在边界上自动转换：

- robot `observe/reset`：native frame -> embodia `Frame`
- policy `infer`：embodia `Frame` -> native frame
- policy 输出：native action -> embodia `Action`
- robot `act`：embodia `Action` -> native action

所以最简单的理解就是：YAML 定义标准结构；你的方法要么直接说这套结构，要么让
`MODALITY_MAPS` 告诉 embodia 如何翻译。

标准 action 结构现在是：

```python
{
    "commands": [
        {
            "target": "arm",
            "kind": "cartesian_pose_delta",
            "value": [...],
            "ref_frame": "tool",
        },
        {
            "target": "gripper",
            "kind": "gripper_position",
            "value": [0.5],
        },
    ],
    "dt": 0.1,
}
```

现在的 `Action` 是一个按组件组织的命令容器。gripper、hand、suction
这类末端执行器都是一等 robot 组件，不再作为临时附加通道塞进一个扁平动作向量里。

最小的本地推理路径就是：

```python
result = em.run_step(robot, policy)
```

如果你需要异步推理或其他推理期能力，仍然保持同一个入口，只是加一个 runtime：

```python
runtime = em.InferenceRuntime(
    mode=em.InferenceMode.ASYNC,
    overlap_ratio=0.2,
)

result = em.run_step(robot, policy, runtime=runtime)
```

对普通使用者来说，到这里就够了。`check_*`、`from_config(...)` 等低层工具仍然保留，但它们是可选集成工具，不是主路径。

## 示例

`examples/` 固定为四条核心路径：

1. [`examples/01_sync_inference.py`](./examples/01_sync_inference.py)
2. [`examples/02_async_inference.py`](./examples/02_async_inference.py)
3. [`examples/03_data_collection.py`](./examples/03_data_collection.py)
4. [`examples/04_replay_collected_data.py`](./examples/04_replay_collected_data.py)

它们共用 [`examples/basic_runtime.yml`](./examples/basic_runtime.yml)。
这个共享配置定义了 `arm` 和 `gripper` 两个组件，Python 示例里每一步也都会输出对应的 `Action.commands`。

## 设计

embodia 的中心始终是统一运行时数据流。最核心的对象是 `Frame`、`Action`、`RobotProtocol`、`PolicyProtocol`、`RobotMixin`、`PolicyMixin`、`run_step()` 和 `InferenceRuntime`。推荐的边界是：robot 和 policy 继续做自己的 native 工作，embodia 负责外围的对齐、映射、校验和运行时流转。

如果你需要更多细节，可以继续看 [`docs/mixin_guide.md`](./docs/mixin_guide.md)、[`docs/yaml_config_example.yml`](./docs/yaml_config_example.yml) 和 [`docs/examples_guide.md`](./docs/examples_guide.md)。
