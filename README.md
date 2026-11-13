# wanxiang-autotrain

万象（Wanxiang）自动化训练流水线 —— 从零训练自有大模型的训练基础设施。

## 当前状态：M1 Trainer Runner ✅ / M1.5 评测器 + 导出契约 ✅

### M1 交付（已通过 55 项测试）

- **配置系统**（`autotrain/config.py`）：YAML 配置 + 确定性哈希守卫。`max_steps` 作为预算旋钮不参与哈希，延长训练预算可合法续训；其余字段变更拒绝续训（防脏实验）
- **基座模型**（`autotrain/trainer/model.py`）：GPT 式 decoder-only，RMSNorm + SwiGLU + 权重绑定，100M 级
- **数据加载**（`autotrain/trainer/data.py`）：uint16 token 流 + 游标续训，train/val 尾部分割
- **checkpoint**（`autotrain/trainer/checkpoint.py`）：原子落盘，含模型/优化器/调度器/数据游标/三份 RNG 状态；恢复时校验配置哈希
- **监控**（`autotrain/trainer/monitor.py`）：3σ loss spike 检测（预热期不误报）+ val loss 停滞 early-stop
- **训练循环**（`autotrain/trainer/runner.py`）：warmup + cosine 调度、梯度累积到 global_batch_tokens、spike 自动回滚降 LR
- **实验存储**（`autotrain/store/experiment_store.py`）：SQLite 实验记录 + 指标时序

### M1.5 交付（只读评测器 + 自描述导出契约）

M1 只有 `.pt` checkpoint，没有「训练产物能被别人读」和「改动到底有没有变好」这两件事。M1.5 补上这两条，插在 M2 数据管线之前，作为后续所有自我改进回路的测量仪器。

| 组件 | 位置 | 作用 |
|------|------|------|
| 导出契约 | `autotrain/export/contract.py` | `wanxiang-m15-1`：bundle 文件集、config schema、逐文件 checksum、逐张量 sha256、tied alias |
| 导出器 | `autotrain/export/exporter.py` | `.pt` → bundle，临时目录 + 校验 + 原子改名；失败绝不留半包 |
| 加载器 | `autotrain/export/loader.py` | bundle → `WanxiangModel`，`cuda → mps → cpu`，不依赖训练循环 |
| 基线导出 | `autotrain/export/exporter.py::export_initial` | 合成确定性 step-0 权重，让第一次训练就有可比的 A/B |
| 采样观察 | `autotrain/export/sampler.py` | 最小 greedy 生成（无 KV cache / batching），仅用于观察 |
| 冻结评测集 | `evals/*.bin` + `manifest.json` | 200K tokens 结构化 Markov 集 + 200K tokens i.i.d. 均匀负对照，校验和入库 |
| 评测规格 | `autotrain/eval/evalspec.py` | 全部阈值/窗口/种子为常量，`evalspec_hash` 封锁；`verify_corpus()` 只读校验 |
| 客观指标 | `autotrain/eval/metrics.py` | 交叉熵、bits/token、perplexity、结构捕获率、greedy 转移命中率；无 LLM-as-a-judge |
| 评测器 | `autotrain/eval/runner.py` | bundle → 封印报告；A/B 判定 `IMPROVED / REGRESSED / NO_CHANGE / INCOMPARABLE` |

**为什么用 Markov 语料**：其 unigram 分布仍是均匀的，理论最优交叉熵是 `ln(8192)=9.0109`，而真实一阶转移分布的最优值约 `2.30` nats。低于 `ln(V)` 只可能来自「学到转移结构」，不可能来自「背词频」。i.i.d. 均匀集是负对照：如果模型在那里也显得变好，说明测量坏了。

**只读边界**：训练代码不写 bundle，评测器不写 `evals/`；任何阈值、窗口、种子或语料字节的变动都会改变 `evalspec_hash`，使新旧报告直接判为 `INCOMPARABLE`，从而杜绝「自己给自己挪球门」。`COMPAT_KEYS` 还包含 `device`——CPU / MPS / CUDA 的 logits 不是逐位一致的，跨设备 A/B 不会被误判为真实提升。

#### Bundle 结构

```
<bundle>/
  config.json        架构、结构、tokenizer 元数据、dtype
  model.safetensors  权重（省略 tied alias `lm_head.weight`）
  tokenizer.json     最小 token-id 规格（M1 是 uint16 flat 流，M2 替换）
  MANIFEST.json      来源、step、逐张量清单、逐文件 checksum
```

#### CLI

```bash
# 训练前：确定性 step-0 基线
python -m autotrain.export export-initial \
  --config autotrain/configs/wanxiang-smoke-4m.yaml --out runs/smoke-4m/bundle_step0

# 训练后：导出 + 与源 checkpoint 逐位校验
python -m autotrain.export export \
  --checkpoint runs/smoke-4m/checkpoints/ckpt_step_500.pt \
  --config autotrain/configs/wanxiang-smoke-4m.yaml --out runs/smoke-4m/bundle
python -m autotrain.export validate --bundle runs/smoke-4m/bundle \
  --checkpoint runs/smoke-4m/checkpoints/ckpt_step_500.pt

# 冻结语料校验 / 同分布不同 seed 的训练语料
python -m autotrain.eval verify
python -m autotrain.eval make-corpus --out runs/train_tokens.bin --tokens 2000000

# 评测与 A/B（两次必须用同一 --device）
python -m autotrain.eval run --bundle runs/smoke-4m/bundle_step0 --device cpu --out runs/report_step0.json
python -m autotrain.eval run --bundle runs/smoke-4m/bundle       --device cpu --out runs/report_trained.json
python -m autotrain.eval compare runs/report_step0.json runs/report_trained.json
```

完整验证顺序见 `VERIFY.md`。按仓库约定：代码与测试已写完，构建与测试由使用者自行执行。

## 快速开始

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # 必须 editable：evals/ 是仓库级冻结资产，不进 wheel
pytest tests/ -v
```

### Windows 一键脚本

`scripts/verify_windows.ps1`（PowerShell 5.1、纯 CPU、幂等）封装了完整验证流程：

```powershell
cd D:\wanxiang-autotrain
powershell -ExecutionPolicy Bypass -File .\scripts\verify_windows.ps1
# 追加完整 M1.5 A/B（含短训练，明显更慢）：
powershell -ExecutionPolicy Bypass -File .\scripts\verify_windows.ps1 -WithTraining
```

脚本自动建 Python 3.12 venv、装 CPU 版 torch 与 editable 包、校验冻结语料、跑全量 pytest、自检导出契约；`-WithTraining` 再跑结构化语料 → 训练 → 导出 → 两次评测 → A/B。产物全在 `runs/windows-verify/`，不修改仓库跟踪文件、不提交、不推送。细节见 `VERIFY.md` 附录 A。

端到端冒烟（4M 配置 500 步，CPU 几分钟）：

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
| M1.5 | 只读评测器 + 自描述导出契约 | ✅ |
| M2 | 数据管线（采集/清洗/配比/版本化） | 待开始 |
| M3 | Orchestrator + 一键流水线 | 待开始 |
| M4 | 超参搜索（代理模型扫描 + μP 外推） | 待开始 |

M1.5 之后才做 M2 的原因：真实数据一旦进来，评测信号会被数据分布变化污染，先要有「与训练代码解耦、自身不可被修改」的基准，才能判断后续每一次改动是进步还是噪声。

设计文档见 `autotrain-design/`（在主工作区，未纳入本仓库）。

## 背景

万象是 OSGame（王者万象棋）项目的自有大模型：玩家侧承担复盘教练、机器人训练器、棋局语义层；开发者侧承担日常开发助手。详见项目设计文档。
