# wanxiang-autotrain

万象（Wanxiang）自动化训练流水线 —— 从零训练自有大模型的训练基础设施。

## 当前状态：M1 Trainer Runner ✅

M1 交付（已通过 55 项测试）：

- **配置系统**（`autotrain/config.py`）：YAML 配置 + 确定性哈希守卫。`max_steps` 作为预算旋钮不参与哈希，延长训练预算可合法续训；其余字段变更拒绝续训（防脏实验）
- **基座模型**（`autotrain/trainer/model.py`）：GPT 式 decoder-only，RMSNorm + SwiGLU + 权重绑定，100M 级
- **数据加载**（`autotrain/trainer/data.py`）：uint16 token 流 + 游标续训，train/val 尾部分割
- **checkpoint**（`autotrain/trainer/checkpoint.py`）：原子落盘，含模型/优化器/调度器/数据游标/三份 RNG 状态；恢复时校验配置哈希
- **监控**（`autotrain/trainer/monitor.py`）：3σ loss spike 检测（预热期不误报）+ val loss 停滞 early-stop
- **训练循环**（`autotrain/trainer/runner.py`）：warmup + cosine 调度、梯度累积到 global_batch_tokens、spike 自动回滚降 LR
- **实验存储**（`autotrain/store/experiment_store.py`）：SQLite 实验记录 + 指标时序

## 快速开始

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

端到端冒烟（4M 模型 500 步，CPU 几分钟）：

```bash
python -c "
import numpy as np
from autotrain.trainer.data import TokenDataset
rng = np.random.default_rng(7)
TokenDataset.write_token_file('smoke_tokens.bin', rng.integers(0, 8192, size=2_000_000, dtype=np.uint16))
"
python -m autotrain.trainer \
  --config autotrain/configs/wanxiang-smoke-4m.yaml \
  --data smoke_tokens.bin \
  --out runs/smoke-4m \
  --db runs/experiments.db
```

## 路线图

| 里程碑 | 内容 | 状态 |
|--------|------|------|
| M1 | Trainer Runner + 断点续训 | ✅ |
| M2 | 数据管线（采集/清洗/配比/版本化） | 待开始 |
| M3 | Orchestrator + 一键流水线 | 待开始 |
| M4 | 超参搜索（代理模型扫描 + μP 外推） | 待开始 |

设计文档见 `autotrain-design/`（在主工作区，未纳入本仓库）。

## 背景

万象是 OSGame（王者万象棋）项目的自有大模型：玩家侧承担复盘教练、机器人训练器、棋局语义层；开发者侧承担日常开发助手。详见项目设计文档。
