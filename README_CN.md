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

如果你想使用可选的远程策略能力：

```bash
pip install ".[remote]"
```

`embodia` 现在以 `numpy` 为核心数值后端，`image/state/action` 里的张量都会在
core 里统一成 `numpy.ndarray`。如果你的项目已经在用 `torch`，embodia 也可以在
运行时边界接收 tensor，然后转换成 numpy 核心结构。

## 快速开始

把 embodia 放在你现有类的最外层，原来的 native 方法保持不变：
下面凡是带 `YOUR_OWN_` 前缀的名字，都只是提示你这里需要自己替换：

```python
import embodia as em


class YourRobot(em.RobotMixin):
    def YOUR_OWN_get_obs(self): ...
    def YOUR_OWN_send_action(self, action): ...
    def YOUR_OWN_reset(self): ...


class YourPolicy(em.PolicyMixin):
    def YOUR_OWN_clear_state(self): ...
    def YOUR_OWN_infer(self, frame): ...
```

然后从 YAML 加载接口对齐配置：

```python
robot = YourRobot.from_yaml("docs/yaml_config_example.yml")
policy = YourPolicy.from_yaml("docs/yaml_config_example.yml")
```

这个 YAML 只描述共享 schema 和方法别名，不承载构造参数。对 policy 来说，
embodia 会直接从 `schema:` 推导输入输出。如果模型需要额外条件输入，比如
`YOUR_OWN_prompt`，放到 `Frame.task` 里。robot spec 不再声明 task 相关能力。

### YAML 和你的方法到底是什么关系

`schema:` 定义的是标准运行时字段名，`method_aliases:` 只负责告诉 embodia
你现有类里的哪个方法对应 `observe`、`act`、`reset`、`infer`。它不会帮你生成
实现，也不会改你的构造函数。

如果你没有额外声明 Python 侧的 `MODALITY_MAPS`，那你的 native 方法就应该直接
使用 YAML 里写的那些名字。也就是说，YAML 本身就是运行时契约，embodia 会按它来
校验输入输出。

对应关系是：

- `schema.images` -> `frame.images` 里应该出现的键
- `schema.components.<name>` -> 一个共享组件键，会同时出现在
  `frame.state[<name>]` 和 `action.commands[<name>]` 里
- `schema.components.<name>.command` -> 这个组件允许使用的
  `Command.command` 值
- `schema.task` -> 仅 policy 侧使用的 `frame.task` 键

凡是你需要在自己项目里替换的 schema 键，embodia 的示例和文档都会写成
`YOUR_OWN_*`。
当前 schema 里已经没有单独的 `Command.target` 字段了，目标组件就是
`Action.commands` 上的那个字典键。

### 方法输入输出

你的 native 方法只需要围绕一份标准 `obs` 和一份标准 `action` 来写：

```python
import numpy as np

obs = {
    # 可选，不填时由 embodia 自动补
    "timestamp_ns": 0,
    "sequence_id": 0,
    "images": {
        "YOUR_OWN_front_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
    },
    "state": {
        "YOUR_OWN_left_arm": np.zeros(6, dtype=np.float32),
        "YOUR_OWN_left_gripper": np.array([0.5], dtype=np.float32),
        "YOUR_OWN_right_arm": np.zeros(6, dtype=np.float32),
        "YOUR_OWN_right_gripper": np.array([0.5], dtype=np.float32),
    },
    # 可选，只给 policy 用
    "task": {
        "YOUR_OWN_prompt": "fold the cloth",
    },
}

action = {
    "YOUR_OWN_left_arm": {
        "command": "cartesian_pose_delta",
        "value": np.zeros(6, dtype=np.float32),
    },
    "YOUR_OWN_left_gripper": {
        "command": "gripper_position",
        "value": np.array([0.5], dtype=np.float32),
    },
    "YOUR_OWN_right_arm": {
        "command": "cartesian_pose_delta",
        "value": np.zeros(6, dtype=np.float32),
    },
    "YOUR_OWN_right_gripper": {
        "command": "gripper_position",
        "value": np.array([0.5], dtype=np.float32),
    },
}
```

方法契约就是：

- `YOUR_OWN_get_obs()` / `observe()` -> 返回 `em.Frame` 或这种 `obs` 风格的 `dict`
- robot 的 `YOUR_OWN_reset()` / `reset()` -> 返回结构和 `observe()` 一样
- `YOUR_OWN_infer(frame)` / `infer(frame)` -> 输入一个 `em.Frame`，返回 `em.Action` 或这种
  `action` 风格的 `dict`
- `YOUR_OWN_send_action(action)` / `act(action)` -> 输入一个 `em.Action`
- policy 的 `YOUR_OWN_clear_state()` / `reset()` -> 返回值会被忽略

数值载体默认都应该是 numpy。如果 robot 没返回 `timestamp_ns` 或
`sequence_id`，embodia 会自动补上。

如果你的原项目里名字不是这些标准名，那就把映射保留在 Python 代码里，用
`MODALITY_MAPS` 处理。embodia 会在边界上自动转换：

- robot `observe/reset`：native frame -> embodia `Frame`
- policy `infer`：embodia `Frame` -> native frame
- policy 输出：native action -> embodia `Action`
- robot `act`：embodia `Action` -> native action

所以最简单的理解就是：YAML 定义标准结构；你的方法要么直接说这套结构，要么让
`MODALITY_MAPS` 告诉 embodia 如何翻译。

现在的 `Action` 是一个按组件组织的命令容器。gripper、hand、suction
这类末端执行器都是一等 robot 组件，不再作为临时附加通道塞进一个扁平动作向量里。
真正的运行频率由 `InferenceRuntime` / `RealtimeController` 管，`Action`
本身不再携带单独的 `dt` 字段。如果 action 级还需要 `meta`，embodia 会自动
切回 `{"commands": ..., "meta": ...}` 这种包裹形式。

最小的本地推理路径就是：

```python
result = em.run_step(robot, source=policy)
```

如果你需要异步推理或其他推理期能力，仍然保持同一个入口，只是加一个 runtime：

```python
runtime = em.InferenceRuntime(
    mode=em.InferenceMode.ASYNC,
    overlap_ratio=0.2,
)

result = em.run_step(robot, source=policy, runtime=runtime)
```

之所以更推荐 `source=` 这个名字，是因为第二个输入不一定是本地 policy，
也可能是远端 policy client、一个实现了 `next_action(frame)` 的遥操对象，
或者一个普通 callable。`policy=` 仍然保留作为兼容别名。embodia 的边界现在
更明确：`robot` 只负责本地执行，remote 部署放在 source/policy 这一侧。
如果某个 robot 类自己也能产生命令，也可以直接写
`run_step(robot, source=robot)`。
对 embodia 自己的 remote 传输来说，`RemotePolicy(...)` 现在只需要连接参数，
动作解析会直接从远端响应或服务端 metadata 里自动推断，不需要你在本地再重复
填写 schema 相关字段。

单步 timing 现在属于 embodia 内部运行时细节，不再作为
`run_step(...)` 的公开输出。如果你确实想做时延分析来估计异步参数，
使用 `profile_sync_inference(...)` 即可。

如果远端是 OpenPI policy server，外层调用方式也不用变，只需要在
source 边界做请求/响应适配：

```python
from embodia.contrib import remote as em_remote

source = em_remote.RemotePolicy(
    host="127.0.0.1",
    port=8000,
    openpi=True,
)

result = em.run_step(robot, source=source)
```

默认的 OpenPI 路径里，embodia 会在后台根据包装后的 robot spec 自动推断
请求/响应适配。如果你更想直接复用 OpenPI 官方 client，也可以通过
`runner=...` 传进来，只要它暴露 `infer(obs)` 即可。

对普通使用者来说，到这里就够了。`check_*`、`from_config(...)` 等低层工具仍然保留，但它们是可选集成工具，不是主路径。

## 示例

`examples/` 固定为四条核心路径：

1. [`examples/01_sync_inference.py`](./examples/01_sync_inference.py)
2. [`examples/02_async_inference.py`](./examples/02_async_inference.py)
3. [`examples/03_data_collection.py`](./examples/03_data_collection.py)
4. [`examples/04_replay_collected_data.py`](./examples/04_replay_collected_data.py)

另外还有一组可选的 remote 示例：

5. [`examples/remote/serve_embodia_policy.py`](./examples/remote/serve_embodia_policy.py)
6. [`examples/remote/robot_with_embodia_remote_policy.py`](./examples/remote/robot_with_embodia_remote_policy.py)

它们共用 [`examples/basic_runtime.yml`](./examples/basic_runtime.yml)。
这个共享配置定义了 `YOUR_OWN_arm` 和 `YOUR_OWN_gripper` 两个占位组件，
Python 示例里每一步也都会输出对应的 `Action.commands` 映射。

## 设计

embodia 的中心始终是统一运行时数据流。最核心的对象是 `Frame`、`Action`、`RobotProtocol`、`PolicyProtocol`、`RobotMixin`、`PolicyMixin`、`run_step()` 和 `InferenceRuntime`。推荐的边界是：robot 和 policy 继续做自己的 native 工作，embodia 负责外围的对齐、映射、校验和运行时流转。

如果你需要更多细节，可以继续看 [`docs/mixin_guide.md`](./docs/mixin_guide.md)、[`docs/yaml_config_example.yml`](./docs/yaml_config_example.yml) 和 [`docs/examples_guide.md`](./docs/examples_guide.md)。
