[![中文](https://img.shields.io/badge/中文-简体-blue)](./README_CN.md)  
[![English](https://img.shields.io/badge/English-English-green)](./README.md)

# inferaxis

`inferaxis` 是一个面向 embodied control 的统一数据接口、动态自适应延迟推理系统。
它把 observation 统一成 `Frame`，把 action 统一成 `Action`，并通过
`run_step(...)` 和 `InferenceRuntime(...)` 维持稳定的外层执行方式。

这个项目的核心目标很直接：只要你的数据对齐到同一套运行时接口，外层就可以用同一条
loop 支持：

- 普通同步推理
- 异步 chunk 推理
- 动态自适应延迟的 chunk 调度
- 本地数采
- 采集动作 replay
- 同步推理延迟 profiling 与 runtime 推荐

`inferaxis` 不是机器人中间件，也不是传输层或部署系统。它聚焦的是推理侧的数据契约和
控制闭环。

## 安装

```bash
git clone https://github.com/zywang03/inferaxis.git
cd inferaxis
pip install .
```

`inferaxis` 的 core 统一使用 `numpy.ndarray` 作为数值载体，图像、状态和动作都会
在运行时边界被归一化成 numpy。

## 核心 API

公开表面刻意保持很小：

- `Frame`
- `Action`
- `Command`
- `run_step(...)`
- `InferenceRuntime(...)`
- `ActionEnsembler`
- `ActionInterpolator`
- `RealtimeController`

运行时调用边界是：

- `observe_fn() -> Frame`
- `act_fn(action) -> Action | None`
- `act_src_fn(frame, request) -> Action | list[Action]`

如果只返回一个 `Action`，就等价于 chunk size 为 `1`。如果返回 `list[Action]`，
同一套 source 就可以直接参与 overlap-aware 的异步调度。

## 快速开始

```python
import inferaxis as infra
import numpy as np


class YourExecutor:
    def get_obs(self):
        return infra.Frame(
            images={"front_rgb": np.zeros((2, 2, 3), dtype=np.uint8)},
            state={
                "left_arm": np.zeros(6, dtype=np.float64),
                "left_gripper": np.array([0.5], dtype=np.float64),
                "right_arm": np.zeros(6, dtype=np.float64),
                "right_gripper": np.array([0.5], dtype=np.float64),
            },
        )

    def send_action(self, action):
        return action


class YourPolicy:
    def infer(self, frame, request):
        del frame, request
        return infra.Action(
            commands={
                "left_arm": infra.Command(
                    command=infra.BuiltinCommandKind.CARTESIAN_POSE_DELTA,
                    value=np.zeros(6, dtype=np.float64),
                ),
                "left_gripper": infra.Command(
                    command=infra.BuiltinCommandKind.GRIPPER_POSITION,
                    value=np.array([0.5], dtype=np.float64),
                ),
                "right_arm": infra.Command(
                    command=infra.BuiltinCommandKind.CARTESIAN_POSE_DELTA,
                    value=np.zeros(6, dtype=np.float64),
                ),
                "right_gripper": infra.Command(
                    command=infra.BuiltinCommandKind.GRIPPER_POSITION,
                    value=np.array([0.5], dtype=np.float64),
                ),
            }
        )


executor = YourExecutor()
policy = YourPolicy()

result = infra.run_step(
    observe_fn=executor.get_obs,
    act_fn=executor.send_action,
    act_src_fn=policy.infer,
)
```

如果你只想做标准化后的 `frame -> action` 推理，不想本地执行动作：

```python
result = infra.run_step(
    frame=my_frame,
    act_src_fn=policy.infer,
    execute_action=False,
)
```

## 数据接口

`Frame` 是统一后的 observation 容器：

```python
frame = infra.Frame(
    images={"front_rgb": np.ndarray(...)},
    state={
        "left_arm": np.ndarray(...),
        "left_gripper": np.ndarray(...),
        "right_arm": np.ndarray(...),
        "right_gripper": np.ndarray(...),
    },
)
```

`Action` 是统一后的控制容器：

```python
action = infra.Action(
    commands={
        "left_arm": infra.Command(
            command=infra.BuiltinCommandKind.CARTESIAN_POSE_DELTA,
            value=np.ndarray(...),
        ),
        "left_gripper": infra.Command(
            command=infra.BuiltinCommandKind.GRIPPER_POSITION,
            value=np.ndarray(...),
        ),
    },
)
```

运行时规则可以概括成：

- `observe_fn()` 必须返回 `inferaxis.Frame`
- `act_src_fn(frame, request)` 必须返回 `inferaxis.Action` 或 `list[inferaxis.Action]`
- `act_fn(action)` 接收 `inferaxis.Action`
- `timestamp_ns` 和 `sequence_id` 由 inferaxis 在后台统一生成

`command` 不是任意字符串，它必须匹配该 component 声明的 command kind。内置 kind
包括：

- `joint_position`
- `joint_position_delta`
- `joint_velocity`
- `cartesian_pose`
- `cartesian_pose_delta`
- `cartesian_twist`
- `gripper_position`
- `gripper_position_delta`
- `gripper_velocity`
- `gripper_open_close`
- `hand_joint_position`
- `hand_joint_position_delta`
- `eef_activation`

如果你有项目私有动作类型，也可以注册成 `custom:...`。

## Runtime 能力

`run_step(...)` 是唯一的外层 loop 入口。`InferenceRuntime(...)` 负责在不改变这条
外层调用方式的前提下，补充调度和优化能力。

```python
runtime = infra.InferenceRuntime(
    mode=infra.InferenceMode.ASYNC,
    overlap_ratio=0.5,
    action_optimizers=[
        infra.ActionEnsembler(current_weight=0.5),
        infra.ActionInterpolator(steps=1),
    ],
    realtime_controller=infra.RealtimeController(hz=50.0),
)

result = infra.run_step(
    observe_fn=executor.get_obs,
    act_fn=executor.send_action,
    act_src_fn=policy.infer,
    runtime=runtime,
)
```

这样同一套数据接口就能支持：

- 同步或异步的 chunk 执行
- 基于 overlap 的异步 chunk 调度
- 通过 `ActionEnsembler(...)` 做 chunk handoff 融合
- 通过 `ActionInterpolator(...)` 做逐步插值
- 带节拍控制的闭环执行
- `profile_sync_inference(...)` 基于目标控制频率的延迟 profiling
- `recommend_inference_mode(...)` 模式推荐

当 `enable_rtc=True` 时，传给 `policy.infer(...)` 的请求对象会直接提供
`request.prev_action_chunk`、`request.inference_delay` 和
`request.execute_horizon`。同一组值也会镜像到 `request.rtc_args`
里，方便按对象整体访问：

- `prev_action_chunk`：当前正在执行的完整 active chunk 快照，不会随着 live buffer 变短而缩短
- `inference_delay`：RTC 应该从这个 chunk 的哪个索引开始生效，计算方式是 `已执行前缀步数 + max(当前估计延迟步数, 1)`
- `execute_horizon`：`prev_action_chunk` 的长度，因此 RTC 的有效区间是 `[inference_delay, execute_horizon)`

对于 chunk 异步执行，inferaxis 现在使用：

- `overlap_steps = floor(overlap_ratio * chunk_size)`
- `trigger_steps = ceil(H_hat) + overlap_steps`

其中 `H_hat` 是按控制步数直接做 EMA 的请求延迟估计。结果返回后，
inferaxis 会先丢掉已经过期的前缀；如果启用了 `ActionEnsembler(...)`，
就对 overlap 区段里同一未来时间步的旧/新 action 做融合，否则直接切到
新的对齐后 chunk。`current_weight` 可以是一个标量，表示整个 overlap 都用
同一个新 chunk 权重；也可以是 `(low, high)`，表示从最早的 overlap step
线性过渡到最晚的 overlap step。对内置 gripper command，inferaxis 会直接
切到新 chunk，不会做数值平均。`ActionEnsembler(current_weight=...)`
不会再对每一步输出额外做一层 temporal filter。也就是说，inferaxis 不是靠
预先写死的
固定时序在跑，而是会根据实际测到的 chunk 延迟在线调整请求时机，因此它本质上
是一个动态自适应延迟推理系统。

## 校验

`check_policy(...)` 和 `check_pair(...)` 是 dry-run 校验工具。

- 它们只检查接口契约
- 最多请求一次 observation 和一次 policy inference
- 不会调用 `act_fn(...)`

## 示例

公开示例固定为以下五个：

1. [`examples/01_sync_inference.py`](./examples/01_sync_inference.py)
2. [`examples/02_async_inference.py`](./examples/02_async_inference.py)
3. [`examples/03_data_collection.py`](./examples/03_data_collection.py)
4. [`examples/04_replay_collected_data.py`](./examples/04_replay_collected_data.py)
5. [`examples/05_profile_inference_latency.py`](./examples/05_profile_inference_latency.py)
6. [`examples/06_async_inference_with_rtc.py`](./examples/06_async_inference_with_rtc.py)

这五个例子一起表达的就是这个项目现在的边界：一套统一数据接口，一条统一外层 loop，
覆盖多种推理时场景。

更细一点的说明见 [`docs/plain_objects_guide.md`](./docs/plain_objects_guide.md)
和 [`docs/examples_guide.md`](./docs/examples_guide.md)。
