[![中文](https://img.shields.io/badge/中文-简体-blue)](./README_CN.md)  
[![English](https://img.shields.io/badge/English-English-green)](./README.md)

# embodia

`embodia` 是一个面向机器人和模型的轻量 Python 运行时接口库。它只解决一个核心问题：让不同项目里的 robot 类和 model 类说同一种数据流语言。它不做训练框架，不做服务系统，不依赖 ROS，也不试图变成大而全平台。

对大多数用户来说，真正需要接触的表面应该很小：

1. 继承 `RobotMixin` / `ModelMixin`
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


class YourModel(em.ModelMixin):
    def clear_state(self): ...
    def infer(self, frame): ...
```

然后从 YAML 加载接口对齐配置：

```python
robot = YourRobot.from_yaml("docs/yaml_config_example.yml")
model = YourModel.from_yaml("docs/yaml_config_example.yml")
```

这个 YAML 只描述接口对齐，不承载构造参数。如果模型需要额外条件输入，比如 prompt，放到 `Frame.task` 里。

标准 action 结构现在是：

```python
{
    "mode": "ee_delta",
    "value": [...],                 # 主动作向量
    "channels": {"gripper": 0.5},   # 可选命名附加执行器
    "ref_frame": "tool",
    "dt": 0.1,
}
```

这里的 `channels` 是通用扩展位，`gripper` 只是一个常见 key，不是写死在 schema 里的专用字段。

最小的本地推理路径就是：

```python
result = em.run_step(robot, model)
```

如果你需要异步推理或其他推理期能力，仍然保持同一个入口，只是加一个 runtime：

```python
runtime = em.InferenceRuntime(
    mode=em.InferenceMode.ASYNC,
    overlap_ratio=0.2,
)

result = em.run_step(robot, model, runtime=runtime)
```

对普通使用者来说，到这里就够了。`check_*`、`from_config(...)` 等低层工具仍然保留，但它们是可选集成工具，不是主路径。

## 示例

`examples/` 固定为四条核心路径：

1. [`examples/01_sync_inference.py`](./examples/01_sync_inference.py)
2. [`examples/02_async_inference.py`](./examples/02_async_inference.py)
3. [`examples/03_data_collection.py`](./examples/03_data_collection.py)
4. [`examples/04_replay_collected_data.py`](./examples/04_replay_collected_data.py)

它们共用 [`examples/basic_runtime.yml`](./examples/basic_runtime.yml)。
这个共享配置已经把 `joint_positions` 和 `gripper_position` 都放进了 state，
而 Python 示例里的末端附加执行器则统一通过 `Action.channels` 表达。

## 设计

embodia 的中心始终是统一运行时数据流。最核心的对象是 `Frame`、`Action`、`RobotProtocol`、`ModelProtocol`、`RobotMixin`、`ModelMixin`、`run_step()` 和 `InferenceRuntime`。推荐的边界是：robot 和 model 继续做自己的 native 工作，embodia 负责外围的对齐、映射、校验和运行时流转。

如果你需要更多细节，可以继续看：

- [`docs/mixin_guide.md`](./docs/mixin_guide.md)
- [`docs/yaml_config_example.yml`](./docs/yaml_config_example.yml)
- [`docs/examples_guide.md`](./docs/examples_guide.md)
