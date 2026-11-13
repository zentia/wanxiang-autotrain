# M1 Trainer Runner + M1.5 评测器/导出契约 — 验证步骤

> 按约定：代码与测试已全部写完，构建与测试由你自行执行，本次没有代为运行任何 `pytest`、训练或导出命令。以下命令在本机（macOS / zsh）验证。

## 1. 环境准备

```bash
cd /Users/zentiali/WorkBuddy/2026-11-13-13-02-17/autotrain   # 或你 clone 后的仓库目录

# 建议用 uv 或 venv
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
# 或传统方式:
# python3 -m venv .venv && source .venv/bin/activate
# pip install -e ".[dev]"
```

依赖：torch、numpy、pyyaml、safetensors、pytest。CPU 版 torch 即可跑通全部测试。

**必须用 editable 安装**（`-e`）：`evals/` 是仓库级冻结资产，不在 wheel 里；`verify_corpus()` 按 `__file__` 的仓库根定位它。用 wheel 安装会报 `corpus manifest not found`。

## 2. 跑测试（M1 核心验收）

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

> 注意：上面这段 i.i.d. 均匀数据是 M1 原有的「跑通训练」冒烟，模型在这里**学不到任何结构**（理论最优就是 `ln(8192)`）。要看 loss 真正下降，用第 6 节的结构化语料。

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

---

# 6. M1.5 验证（新增）

## 6.1 冻结语料自检

```bash
python -m autotrain.eval verify
```

预期 `status: OK`，并打印 `evals/manifest.json`。本机实测基线（clone 时应逐字一致，不一致说明语料被改过）：

| 文件 | 大小 | SHA256 |
|------|------|--------|
| `evals/structured_eval.bin` | 400000 B | `7e267802d689fd25fcf5f36a79c923602d01c7c59cc0589f7a33f80936d4ca4e` |
| `evals/random_control.bin` | 400000 B | `5b933fb5da355ced3b75da224412ff219c0426eb04813e9242eed61be450bd28` |

各 200000 个 uint16 token。`verify` 会检查文件集、逐文件 checksum、token 数、manifest 的 `frozen` 标记；往 `evals/` 多塞一个 `.bin` 也会报错。

## 6.2 跑 M1.5 测试

```bash
pytest tests/test_export.py tests/test_eval.py -v
# 最终仍跑全量：
pytest tests/ -v
```

| 测试 | 验收点 |
|------|--------|
| `test_export.py::TestExportRoundTrip` | bundle 文件集、逐位往返、tied embedding 恢复、张量清单 |
| `test_export.py::TestContractEnforcement` | 篡改 config / 张量 / 缺文件 / 覆盖保护全部被拒 |
| `test_export.py::TestInitialBaseline` | step-0 基线确定性、与训练产物不同、导出不污染调用方 RNG |
| `test_export.py::TestAtomicPublish::test_write_failure_leaves_no_partial_bundle` | 写盘失败不留 `.m15tmp` 半包 |
| `test_eval.py::TestFrozenCorpus` | 冻结字节可由规格重生成；篡改/增删/取消 frozen 全被拒；结构化集 unigram 均匀但转移命中率 ≈ 0.8 |
| `test_eval.py::TestObjectiveMetrics` | `ln(8192)=9.0109`、Markov 闭式最优 ≈ 2.30 nats、`p∈{0,1}` 边界、结构捕获率归一化 |
| `test_eval.py::TestEvalSpecSeal` | A/B 四态判定；规格/语料/配置/窗口/设备任一变化 → `INCOMPARABLE` 且不给分 |
| `test_eval.py::TestEvalRunner` | 评测前后 `evals/` 与 bundle 逐文件 hash 不变（只读）；报告含全部封印字段；未训练模型负对照通过 |

## 6.3 训练前：导出 step-0 基线

M1 只在训练中/后写 checkpoint，没有「之前」的工件。用同一套契约合成确定性未训练权重：

```bash
python -m autotrain.export export-initial \
  --config autotrain/configs/wanxiang-smoke-4m.yaml \
  --out runs/smoke-4m/bundle_step0
python -m autotrain.export describe --bundle runs/smoke-4m/bundle_step0
```

预期：`MANIFEST.json` 中 `step: 0`、`source_checkpoint: "<untrained>"`，四个文件齐全。同一种子重复导出到另一个目录，`files` 里的三个 checksum 应完全相同。

## 6.4 用结构化语料训练

```bash
# 与冻结评测集同分布，但 seed 不相交；写到你指定的路径，绝不写 evals/
python -m autotrain.eval make-corpus --out runs/train_tokens.bin --tokens 2000000

python -m autotrain.trainer \
  --config autotrain/configs/wanxiang-smoke-4m.yaml \
  --data runs/train_tokens.bin \
  --out runs/smoke-4m \
  --db runs/experiments.db
```

预期：val loss 从 ~9.01 明显下降，趋向 2.30 附近（该 Markov 分布的理论最优）。`wanxiang-smoke-4m.yaml` 的 `seq_len: 256`，`derived()` 会给出 `micro_batch=32 / grad_accum=8`。

若只想快速验证链路，可直接把 yaml 的 `max_steps` 调小（例如 20）；`max_steps` 不参与 `config_hash`，改它不会触发「配置变更拒绝续训」。此时导出的 checkpoint 名是 `ckpt_step_<N>.pt`，第 6.5 节命令要相应替换。

## 6.5 训练后：导出并逐位校验

```bash
python -m autotrain.export export \
  --checkpoint runs/smoke-4m/checkpoints/ckpt_step_500.pt \
  --config autotrain/configs/wanxiang-smoke-4m.yaml \
  --out runs/smoke-4m/bundle

python -m autotrain.export validate \
  --bundle runs/smoke-4m/bundle \
  --checkpoint runs/smoke-4m/checkpoints/ckpt_step_500.pt
```

预期：`validate` 返回 `status: OK` 与 `tensors_compared`。`model.safetensors` 不含 `lm_head.weight`（tied alias 省略），加载时才恢复。

## 6.6 评测与 A/B

```bash
# 两次必须显式指定同一个 --device
python -m autotrain.eval run --bundle runs/smoke-4m/bundle_step0 --device cpu --out runs/report_step0.json
python -m autotrain.eval run --bundle runs/smoke-4m/bundle       --device cpu --out runs/report_trained.json
python -m autotrain.eval compare runs/report_step0.json runs/report_trained.json
```

判读：

| 字段 | 预期 |
|------|------|
| `status`（step-0） | `BASELINE` 或 `PARTIAL`，`checks.control_consistent: true`，`structure_gain_ratio ≈ 0` |
| `status`（训练后） | `STRUCTURE_CAPTURED`（`structure_gain_ratio ≥ 0.10`） |
| `metrics.structured.bits_per_token` | 训练后应显著低于 `ln(8192)/ln(2) ≈ 13.0` bits |
| `metrics.random_control.deviation_from_uniform_nats` | 两次都应在 ±0.30 内；否则 `status: INVALID`，说明测量坏了 |
| `compare.verdict` | `IMPROVED`，且 `compatible: true`、`delta_bits_per_token ≥ 0.05` |

**Mac 上特别注意**：M1 的 `TrainerRunner` 仍只有 `cuda/cpu` 分支，没有 `mps`，训练会走 CPU；而 M1.5 的 `default_device()` 是 `cuda → mps → cpu`。若一次用默认（mps）、一次用 `--device cpu`，`device` 属于 `COMPAT_KEYS`，`compare` 会直接判 `INCOMPARABLE` 而不是给出分数。这是刻意设计：不同后端的 logits 不是逐位一致的，跨设备差值不能算真实提升。**两次统一 `--device cpu` 最省事**。

## 6.7 篡改自检（可选）

```bash
cp -r evals /tmp/evals_backup
# 改一个字节后，下面命令必须报 checksum mismatch
python -m autotrain.eval verify --eval-dir <被改动的目录>
python -m autotrain.export describe --bundle <被改动 config.json 的 bundle>   # 必须报 checksum mismatch
```

`MANIFEST.json` 记录 `config.json` / `model.safetensors` / `tokenizer.json` 的 checksum，加载时逐一核对；`MANIFEST` 自身不参与哈希以避免自指。

---

## 已知边界（M1 + M1.5 范围内未做）

- 数据管线（采集/清洗/配比）→ M2，当前直接消费 uint16 token 文件
- Orchestrator 状态机与自动重启 → M3，当前手动重跑即续训
- lm-eval-harness 接入 → M4，`EvalConfig` 已预留字段
- loss spike 回滚已实现（检测+回滚+降LR），但自动重启循环属 Orchestrator
- **M1 runner 没有 mps 分支**，Mac 训练走 CPU；M1.5 加载器支持 mps。补 mps 属独立改动，本轮未动 M1 训练代码
- **评测集是合成语料**，只回答「模型能否学到一阶转移结构」，不测语言能力；M2 引入真实数据后应新建 `wanxiang-eval-2` 并重新冻结，旧报告保留且不与新报告混比
- 训练语料与评测语料同分布、不同 seed，属同分布 holdout，不是记忆泄漏测试；若日后要测泄漏，另建独立集
- `greedy_generate` 每步做全序列 forward，无 KV cache / batching，`GEN_TOKENS=64` 在 `seq_len=256` 的 CPU 上会花一点时间，属预期
- `evals/` 不随 wheel 分发，只支持 editable 安装或从仓库根运行

---

# 附录 A：在 Windows 本机运行（无需 Mac）

M1.5 全部代码只用 `pathlib` / `numpy` / `torch` / `safetensors` / `pytest`，没有 Unix 专有调用；冻结语料是显式 `<u2` 小端，`evals/*.bin` 被 git 判为 binary（本机 `core.autocrlf=true` 也不会改写），实测 clone 后 SHA256 与 manifest 一致。**结论：Windows 原生可跑，只是本机目前没有任何 Python 环境装齐 torch / safetensors / pytest，需要先装一次。**

## A.1 环境实测（2026-11-13，本机）

| 环境 | Python | numpy | torch | pyyaml | safetensors | pytest |
|---|---|---|---|---|---|---|
| PATH 默认（WorkBuddy 自带） | 3.13.14 | ✗ | ✗ | ✗ | ✗ | ✗ |
| `C:\Python314` / uv 3.12 / Python 3.10 | 3.14 / 3.12 / 3.10 | 仅 3.10 有 | ✗ | 仅 3.10 有 | ✗ | ✗ |
| WSL `/opt/sglang`（需 `wsl -u root`） | 3.12.16 | ✓ | ✓ (cu130) | ✓ | ✓ | ✗ |

推荐 **Python 3.12**（uv 那个）建独立 venv，避开 3.13/3.14 的 wheel 兼容不确定性。本机代理 `HTTPS_PROXY=http://127.0.0.1:63366` 已注入，pip 可直接走代理。

## A.2 PowerShell 命令序列（推荐，纯 CPU 即可完整验证 M1.5）

```powershell
cd D:\wanxiang-autotrain
& "C:\Users\zentiali\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

.\.venv\Scripts\python.exe -m autotrain.eval verify
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

`pytest tests/ -v` 里的 tiny model 显式走 CPU（`test_eval.py` / `test_export.py` 都传了 `device="cpu"`）；第 6 节的 export / eval run / compare 也统一用 `--device cpu`，A/B 依然可比。

**上面这套命令已封装成脚本**（`scripts/verify_windows.ps1`），不必逐条复制：

```powershell
cd D:\wanxiang-autotrain
powershell -ExecutionPolicy Bypass -File .\scripts\verify_windows.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\verify_windows.ps1 -WithTraining -TrainSteps 50
```

脚本特性：幂等（venv 建好、依赖装齐后跳过 pip）、任一步失败即停并打印失败步骤与退出码、全程 `--device cpu` 保证 A/B 可比；**原生命令的 stderr 已合并进 stdout**——M1 trainer 用 `logging` 写 stderr，PS 5.1 会把每一行包装成 `NativeCommandError`，外层 shell 再报成 `RemoteException` 红色刷屏（进程其实健康，loss 正常下降）。脚本里已消除；若直接手敲命令遇到该提示，在末尾加 `2>&1 | ForEach-Object { $_.ToString() }` 即可。训练用的 `max_steps` 只写在 `runs/windows-verify/smoke_cpu.yaml` 副本里，仓库内 yaml 不动（`max_steps` 本来就不参与 `config_hash`）。默认产物：冻结语料校验输出、pytest 结果、`bundle_step0`；`-WithTraining` 追加 `report_step0.json` / `report_trained.json` / compare 结论。`-Python <path>` 可换解释器，`-SkipInstall` 可完全复用现有 venv。

若要顺带跑 M1 训练并用上 RTX 5080（SM120，需 CUDA ≥12.8 的 torch），按 pytorch.org 选择器改装匹配的 Windows CUDA 版 wheel、不要用 `cpu` index；`TrainerRunner` 的 `cuda if torch.cuda.is_available()` 分支会自动生效，比 Mac 走 CPU 快得多。此时 `default_device()` 会选 cuda，**两次评测仍须显式统一同一个 `--device`**。

## A.3 WSL 备选（零大文件下载）

`/opt/sglang` venv 已装 torch(cu130) / numpy / yaml / safetensors，只差 pytest；其 python 是指向 root 目录的符号链接，需 `wsl -u root`：

```bash
wsl -u root -e bash -lc 'cd /mnt/d/wanxiang-autotrain && \
  /opt/sglang/bin/python -m pip install pytest && \
  /opt/sglang/bin/python -m pip install -e . && \
  /opt/sglang/bin/python -m autotrain.eval verify && \
  /opt/sglang/bin/python -m pytest tests/ -v'
```

注意 drvfs 上 editable 安装与 pytest 收集比原生 NTFS 慢，此前 `git status` 在该挂载上曾超时。若明显卡顿，把仓库连同 `evals/` 一起 `cp -r` 到 WSL 内 ext4 目录再跑（语料 checksum 不变）。

## A.4 Windows 特有注意点

- `exporter` 用 `tmp.replace(out)` 做**目录级**原子改名：先 `shutil.rmtree(out)` 保证目标不存在，Windows 上可行；若 `out` 被资源管理器/杀软占用，`--force` 覆盖可能失败，换一个输出目录即可。
- `.py` / `.md` 会被 `core.autocrlf=true` 转成 CRLF，不影响 Python 与 JSON 解析；`evals/*.bin` 是 binary，不受影响。若要彻底锁死，可加 `.gitattributes`：`evals/*.bin binary`。
- 第 6 节命令是 zsh 写法（反斜杠续行、`source`、正斜杠路径），在 PowerShell 里按 A.2 的写法替换；CLI 参数本身完全一致。

---

# 附录 B：首次 Windows 实跑结果（2026-11-13）

执行 `scripts/verify_windows.ps1 -WithTraining`（先 20 步，再 `-TrainSteps 50` 续到 50 步）。脚本能走到训练并完成，说明前面的 preflight / venv / 依赖 / `eval verify` / **全量 pytest** / step-0 导出与重载检查全部通过；任一步失败脚本都会 `throw` 停下。

## B.1 实验记录（`experiments.db`）

| id | max_steps | 实际步数 | status | 起点 loss | 终点 loss | 终点 lr |
|---|---|---|---|---|---|---|
| 1 | 20 | 1–20 | DONE | 9.090878 | 8.782557 | 0.00042 |
| 2 | 50 | 21–50（续训） | DONE | 8.734094 | 5.435323 | **0.00100** |

两条记录 `config_hash` 完全相同（`b9efcd7e…`），实证了 `max_steps` 作为预算旋钮不触发「配置变更拒绝续训」；`ckpt_step_20.pt → ckpt_step_50.pt` 的续训链路也顺带被真实跑通。最终 `val_loss = 5.255559`。

## B.2 A/B 报告对比（全部 `COMPAT_KEYS` 一致，可比）

| 指标 | step-0 | step-50 | 变化 |
|---|---|---|---|
| `structured.nats_per_token` | 9.0824 | **5.2308** | −3.8516 |
| `structured.bits_per_token` | 13.1031 | **7.5465** | −5.5567 |
| `structured.perplexity` | 8799.07 | **186.94** | 约 47× |
| `structure_gain_ratio` | −0.0107 | **0.5635** | +0.5741 |
| `greedy_transition_accuracy` | 0.0 | **1.0** | +1.0（64/64 步走 `+1` 链） |
| `random_control.nats_per_token` | 9.0829 | 9.2114 | **+0.1285**（不降反升） |
| `deviation_from_uniform_nats` | +0.0719 | +0.2005 | 仍在 ±0.30 内 |
| `status` | `BASELINE` | `STRUCTURE_CAPTURED` | — |

`delta_bits_per_token = 5.5567 ≥ 0.05` → **`compare_reports` 判定 `IMPROVED`**。负对照不降反升而结构化集大幅下降，是「提升来自学到转移结构、不是整体 CE 漂移或测量坏了」的双向证据；`greedy_transition_accuracy = 1.0` 是不依赖 CE 数值的行为层证据。**M1.5 评测闭环成立。**

## B.3 三个实测发现

1. **训练预算几乎全花在 warmup 上**：`warmup_steps = 50`、`max_steps = 50`，lr 从 `4e-05` 线性升到 `1e-3`（= `lr_peak`）恰好在最后一步，**cosine 衰减从未开始**。当前 5.2308 nats 距理论最优 2.3023 尚差 2.93 nats，只捕获 56.35% 的可学习结构。若要看收敛，应把 `warmup_steps` 降到 10 左右、`max_steps` 提到 200+。

2. **负对照阈值存在理论缺陷（待决策，本轮未改）**：`RANDOM_CONTROL_TOLERANCE_NATS = 0.30` 把 `|deviation| ≤ 0.30` 当作「测量正常」。但一个**完美**学到 Markov 规则的模型，给 `prev+1` 的概率是 `p_hit ≈ 0.8000`、其余各 `p_other = 1/40960`；在 i.i.d. 均匀语料上期望 CE 是
   `-(1/V)·ln(p_hit) - ((V-1)/V)·ln(p_other) ≈ 10.619 nats`，相对 `ln(8192)=9.0109` 即 **deviation ≈ +1.61 nats**，必然越过 0.30 被判 `status: INVALID`。也就是说**学得越好越容易被评测器判为测量失效**。当前 step-50 的 +0.2005 只是尚未越线。
   
   负对照的正确语义应是「不允许它跟着结构化集一起变好」，而不是「必须紧贴 ln(V)」。可选修正：
   - (a) 改成单边判据 `deviation >= -0.30`，允许正向特化；
   - (b) 增加 `control_nats - structured_nats >= margin`，直接检验「结构集显著优于均匀集」；
   - (c) 维持现状但把阈值放宽到 ~1.8 nats（检测力下降，不推荐）。
   
   **注意**：`RANDOM_CONTROL_TOLERANCE_NATS` 参与 `evalspec_hash`，任何改动都会改变封印值，使本附录 B 的两份报告与新规格 `INCOMPARABLE`。若要改，应显式启用 `wanxiang-eval-2` 并重新冻结报告，旧报告作为历史存档保留。建议同时让 `compare_reports` 拒绝 `status == "INVALID"` 的报告。

3. **`compare_reports` 不看 `status`**：即使某份报告是 `INVALID`，只要兼容键一致仍会给出 `IMPROVED/REGRESSED`。这与 (2) 是同一个「评测结果可信度」缺口，建议一并处理。

## B.4 下一步

按优先级：① 决定负对照语义并新建 evalspec 版本；② 调 `warmup_steps` 与训练预算，看能否把 `structure_gain_ratio` 推到接近 1.0；③ 之后才进入 M2 真实数据管线（届时需新建 `wanxiang-eval-2` 并重新冻结，旧报告不混比）。
