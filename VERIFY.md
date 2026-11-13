# M1 Trainer Runner — 验证步骤

> 按约定：代码已全部写完，构建与测试由你自行执行。以下命令在本机（macOS / zsh）验证。

## 1. 环境准备

```bash
cd /Users/zentiali/WorkBuddy/2026-11-13-13-02-17/autotrain

# 建议用 uv 或 venv
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
# 或传统方式:
# python3 -m venv .venv && source .venv/bin/activate
# pip install -e ".[dev]"
```

依赖只有：torch、numpy、pyyaml、pytest。CPU 版 torch 即可跑通全部测试。

## 2. 跑测试（核心验收）

```bash
pytest tests/ -v
```

预期全部通过。关键用例对应设计验收标准：

| 测试 | 验收点 |
|------|--------|
| `test_runner.py::TestResume::test_resume_continues_seamlessly` | **M1 核心验收**：中断后从 checkpoint 续训，跑完剩余步数 |
| `test_runner.py::TestResume::test_resume_rejects_config_change` | 配置哈希守卫：改配置后拒绝续训（防脏实验） |
| `test_checkpoint.py::TestCheckpointRoundTrip` | checkpoint 往返，权重逐位一致 |
| `test_checkpoint.py::TestAtomicWrite` | 落盘原子性，无 .tmp 残留 |
| `test_monitor.py::TestLossSpikeDetector` | loss spike 检测（3σ 判据、预热期不误报） |
| `test_monitor.py::TestEarlyStopTracker` | val loss 停滞 early-stop |
| `test_runner.py::TestModelScale::test_100m_model_parameter_count` | 100M 级模型参数量在 60M~160M 区间 |
| `test_experiment_store.py` | 实验记录与指标时序 CRUD |

## 3. 端到端冒烟（4M 小模型，CPU 几分钟）

```bash
# 生成合成 token 数据（临时脚本）
python -c "
import numpy as np
from autotrain.trainer.data import TokenDataset
rng = np.random.default_rng(7)
TokenDataset.write_token_file('smoke_tokens.bin', rng.integers(0, 8192, size=2_000_000, dtype=np.uint16))
print('2M tokens written')
"

# 一条命令训练
python -m autotrain.trainer \
  --config autotrain/configs/wanxiang-smoke-4m.yaml \
  --data smoke_tokens.bin \
  --out runs/smoke-4m \
  --db runs/experiments.db
```

预期：500 步训练完成，loss 从 ~9 降到明显更低，输出 summary（status/steps/final_val_loss/tokens_per_sec），`runs/smoke-4m/checkpoints/` 下有 ckpt_step_100~500 共 5 个 checkpoint。

## 4. 验证断点续训（手动模拟崩溃）

```bash
# 训练中 Ctrl-C 杀掉进程，然后原命令重跑：
python -m autotrain.trainer \
  --config autotrain/configs/wanxiang-smoke-4m.yaml \
  --data smoke_tokens.bin \
  --out runs/smoke-4m \
  --db runs/experiments.db
```

预期日志出现 `resumed from .../ckpt_step_N.pt at step N`，从 N 继续而不是从 0 开始。

## 5. 查看实验记录

```bash
python -c "
from autotrain.store.experiment_store import ExperimentStore
s = ExperimentStore('runs/experiments.db')
for e in s.find_experiments():
    print(e['id'], e['name'], e['status'])
"
```

## 已知边界（M1 范围内未做，属后续里程碑）

- 数据管线（采集/清洗/配比）→ M2，当前直接消费 uint16 token 文件
- Orchestrator 状态机与自动重启 → M3，当前手动重跑即续训
- lm-eval-harness 接入 → M4，EvalConfig 已预留字段
- loss spike 回滚已实现（检测+回滚+降LR），但自动重启循环属 Orchestrator
