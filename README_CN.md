[![中文](https://img.shields.io/badge/中文-简体-blue)](./README_CN.md)  
[![English](https://img.shields.io/badge/English-English-green)](./README.md)

# embodia

`embodia` 是一个面向机器人和模型的轻量 Python 运行时接口库。它只解决一个核心问题：让不同项目里的 robot 类和 model 类说同一种数据流语言。它不做训练框架，不做网络服务，不做 server 进程，不依赖 ROS，也不试图变成大而全系统。

这个项目的核心边界很清楚：`Protocol` 表达兼容标准，`Mixin` 是低侵入接入方式，`check_*` 是运行时验收入口，`run_step()` 是最小单步数据流原语，`InferenceRuntime` 则在不改变主调用方式的前提下补充 `ActionEnsembler`、`AsyncInference` 和 Hz 控制等推理期能力。`embodia` 默认不负责 episode 边界和持久化格式，这部分交给用户代码或 example 来决定。

## 安装

```bash
git clone https://github.com/zywang03/embodia.git
cd embodia
pip install .
```

如果你希望使用 `from_yaml(...)`，可以安装可选依赖：

```bash
pip install ".[yaml]"
```

如果你需要 OpenPI 远程策略相关能力，可以额外安装：

```bash
pip install ".[openpi-remote]"
```

`embodia` 本身不要求 `numpy` 或 `torch`。如果用户项目里已经安装了这些库，embodia 可以在运行时边界接受它们并归一化成自己的核心数据结构。

## 快速开始

推荐方式是直接修改你项目最外层的 robot / model 类，并把 `RobotMixin` 或 `ModelMixin` 放在最左侧。你保留原有方法，embodia 负责做统一接口、映射、校验和标准化。

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

如果不想把映射配置写死在类里，也可以在构造时注入：

```python
robot = YourRobot.from_config(
    robot_spec={
        "name": "your_robot",
        "action_modes": ["cartesian_delta"],
        "image_keys": ["rgb_front"],
        "state_keys": ["qpos"],
    },
    method_aliases={
        "observe": "capture",
        "act": "send_command",
        "reset": "home",
    },
    modality_maps={
        em.IMAGE_KEYS: {"rgb_front": "front_rgb"},
        em.STATE_KEYS: {"qpos": "joint_positions"},
        em.ACTION_MODES: {"cartesian_delta": "ee_delta"},
    },
)
```

如果你希望把配置完全放到 Python 外面，也可以走 YAML。项目里已经有共享示例配置 [basic_runtime.yml](./examples/basic_runtime.yml) 和说明型配置 [yaml_config_example.yml](./docs/yaml_config_example.yml)。

## 最小数据流

如果你只有机器人和一个动作源，比如遥操作或脚本策略，可以直接用 `run_step()`：

```python
result = em.run_step(robot, action_fn=teleop_or_scripted_policy)
```

如果你有本地模型，同样还是这一套入口：

```python
em.check_pair(robot, model, sample_frame=robot.reset())
result = em.run_step(robot, model)
```

如果你想加推理期优化，比如 action ensemble、异步 chunk 预取或 Hz 控制，也还是在这个入口上叠 runtime：

```python
runtime = em.InferenceRuntime(
    mode=em.InferenceMode.SYNC,
    action_optimizers=[em.ActionEnsembler(window_size=2)],
    realtime_controller=em.RealtimeController(hz=50.0),
)

result = em.run_step(robot, model, runtime=runtime)
```

如果你的策略天然输出 chunk，可以进一步配置 `AsyncInference`，把 overlap / prefetch 调度放在 embodia 后台，而不是塞进 model 包装层里。

```python
runtime = em.InferenceRuntime(
    mode=em.InferenceMode.ASYNC,
    async_inference=em.AsyncInference(
        chunk_provider=my_chunk_provider,
        condition_steps=3,
        prefetch_steps=3,
    ),
    realtime_controller=em.RealtimeController(hz=50.0),
)
```

这里把 `mode` 做成显式字段是刻意的：同步和异步是两种不同的 runtime 语义，不应该只靠有没有传 `async_inference` 来隐式推断。

## 示例

当前 `examples/` 建议固定看这几条主路径：

1. [01_robot_data_collection.py](./examples/01_robot_data_collection.py)：数采脚本
2. [02_rollout_loop.py](./examples/02_rollout_loop.py)：同步推理 / rollout
3. [03_inference_runtime.py](./examples/03_inference_runtime.py)：异步推理与 runtime 能力
4. [04_lerobot_bridge.py](./examples/04_lerobot_bridge.py)：把采集出来的数据 replay / 导出成自定义结构

共享 YAML 配置在 [basic_runtime.yml](./examples/basic_runtime.yml)。

## 设计原则

`embodia` 的重点一直是统一数据流，而不是替用户规定存储格式、训练流程或部署系统。模型尽量只做 `obs -> action`；action 的维护、异步推理、平滑、调度等放进 runtime；多步采集、replay、导出格式交给用户脚本或 example 来定义。这样侵入更小，边界也更稳定。
