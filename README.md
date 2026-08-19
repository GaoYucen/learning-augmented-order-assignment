# ROOD-DASFAA2019

本项目是对以下会议论文的一个干净、可运行的**复现实现**：

> *Real-Time Route Planning and Online Order Dispatch for Bus-Booking Platforms*（DASFAA 2019）

项目采用类似 ST-SACA 的模块化 `src/` 目录结构，但将其中的强化学习组件替换为原论文的问题设定，主要包括：

- 基于历史时段均值的需求预测；
- 考虑容量约束的公交线路构造；
- Random 和 Greedy 在线调度基线；
- Fractional Primal-Dual（FPD）模拟；
- Improved Primal-Dual（IPD）在线整数调度；
- 通过 SciPy MILP 求解的离线混合整数规划上界 `OPT`；
- 原论文报告的五组实验配置；
- 面向成都真实需求数据的 learning-augmented dispatch 实验。

## 重要来源说明

原论文2019年的历史源代码无法获得。本仓库根据正式发表的论文重新构建，仅将 ST-SACA 作为工程结构参考。因此，本项目应描述为**基于论文的复现实现**，不能描述成原始历史源代码或与原作者代码完全一致的实现。

## 环境安装

创建虚拟环境：

```bash
python -m venv .venv
```

Linux/macOS 激活环境：

```bash
source .venv/bin/activate
```

Windows PowerShell 激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

安装项目及依赖：

```bash
pip install -e .
```

也可以不激活环境，直接在 Windows PowerShell 中使用：

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

## 快速开始

运行一组基础实验：

```bash
python scripts/run_experiment.py --experiment 1 --seed 2019
```

运行论文的五组配置并生成 CSV 和图表：

```bash
python scripts/reproduce_paper.py --seed 2019
```

运行测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Learning-augmented 合成 sanity check

该脚本是学生 A 的初始算法验证入口，用于构造可控的 learning-augmented online dispatch 实验：

```text
slot 订单
→ 合成预测
→ predictive LP advice
→ 在线调度算法
→ offline OPT 对比
→ CSV 和图表
```

运行 bottleneck 配置：

```bash
python scripts/synthetic_sanity_check.py --setting bottleneck --slots 5 --prediction-scales 1.0 0.75 1.25 0.5 1.5 --thetas 0.2 0.4 0.6 0.8 --output-dir outputs/synthetic_sanity_bottleneck
```

该脚本比较 `Random`、`Greedy`、`IPD`、`Prediction-only` 和 `RP-LAIPD`。`Prediction-only` 直接遵循 predictive LP 生成的 advice；`RP-LAIPD` 将预测 advice 与原始 IPD 思路结合。`theta` 越大，表示算法越信任预测。

生成文件：

- `synthetic_sanity_results.csv`：每个 slot、方法和预测设置的原始结果；
- `synthetic_sanity_summary.csv`：分组后的均值、标准差和95%置信区间；
- `alg_over_opt_vs_prediction_error.png`：`ALG/OPT` 随预测误差变化的趋势图。

该实验只用于学生 A 的 A1/A2 合成 sanity check，不是最终的真实预测实验。其作用是验证实验链路能够运行，并检查 learning-augmented 方法在预测质量变化时是否表现合理。

核心模块：

- `src/rood_dasfaa2019/learning/request_types.py`：将订单映射为 request type，当前使用 destination station 定义；
- `src/rood_dasfaa2019/learning/prediction.py`：真实预测接入前使用的合成 prediction provider；
- `src/rood_dasfaa2019/learning/metrics.py`：计算 prediction error、advice error、ALG/OPT 和分组汇总；
- `src/rood_dasfaa2019/learning/runner.py`：Random、Greedy、IPD、Prediction-only 和 RP-LAIPD 的统一实验入口。

## 项目目录

```text
configs/                   论文实验和成都实验配置
data/stations/             合成的30站点机场基准
src/rood_dasfaa2019/
  algorithms/              Random、Greedy、FPD、IPD、RP-LAIPD、offline OPT
  data/                    成都数据清洗与准备
  experiments/             成都真实需求实验适配器
  learning/                request type、预测、advice、指标和统一 runner
  routing/                 需求预测和容量感知线路构造
  simulation/              实体、实例生成和模拟指标
  utils/                   配置与可复现工具
scripts/                   数据、训练、实验和绘图入口
outputs/                   生成的 CSV 与图表
tests/                     单元测试和接口测试
```

## 学生 B：成都 B0–B6 实验流水线

真实需求实验使用 `D:/route规划/chengdu order`，冻结配置位于 `configs/chengdu.yaml`。

实验包括：

- B0：冻结数据定义、供给侧、时间切分、随机种子和统一指标；
- B1：成都订单清洗、destination station 构造和 chronological split；
- B2：三种需求预测 baseline 和 decision-aware diagnostics；
- B3：四类 controlled prediction corruption；
- B4：RP-LAIPD、Prediction-only 和 IPD 配对消融；
- B5：请求数 N、车辆数 M、类型数 K 的 runtime/scalability；
- B6：论文图、逐图 CSV 和复现入口。

### B0 冻结设定

```text
时区：Asia/Shanghai
slot：30分钟
destination stations：30
Train：2016-11-01 至 2016-11-20
Validation：2016-11-21 至 2016-11-24
Test：2016-11-25 至 2016-11-27
Shifted Test：2016-11-28 至 2016-11-30
公交数量：10
单车容量：30
RP-LAIPD theta：0.4
随机种子：2026、2027、2028、2029、2030
```

成都订单是真实需求数据，但车辆、容量和固定线路为冻结的半合成供给侧。因此，论文中应将该实验准确表述为：

> 成都真实订单需求与半合成公交供给的 semi-synthetic experiment。

不能将其描述为完整的真实公交运营实验。

### 一键运行

安装项目后，从原始成都订单开始依次运行 B0–B6：

```powershell
.\.venv\Scripts\python.exe scripts\run_chengdu_pipeline.py
```

如果已经存在约610万行的清洗数据、三种 baseline prediction 和 checkpoint prediction，可以跳过数据准备与预测训练：

```powershell
.\.venv\Scripts\python.exe scripts\run_chengdu_pipeline.py --skip-data --skip-prediction
```

也可以单独运行各阶段：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_chengdu_data.py
.\.venv\Scripts\python.exe scripts\train_chengdu_predictions.py
.\.venv\Scripts\python.exe scripts\train_checkpoint_predictions.py
.\.venv\Scripts\python.exe scripts\export_chengdu_supply.py
.\.venv\Scripts\python.exe scripts\run_prediction_diagnostics.py
.\.venv\Scripts\python.exe scripts\run_checkpoint_dispatch.py
.\.venv\Scripts\python.exe scripts\run_distribution_shift.py
.\.venv\Scripts\python.exe scripts\build_ablation_results.py
.\.venv\Scripts\python.exe scripts\run_runtime_scalability.py
.\.venv\Scripts\python.exe scripts\plot_paper_figures.py
.\.venv\Scripts\python.exe scripts\plot_checkpoint_experiment.py
```

### 成都实验输出

主要输出位于 `outputs/chengdu/`：

- `processed/`：B1 清洗订单、实验订单和逐 slot/type 需求；
- `prediction/`：B2 预测明细、checkpoint 预测和预测指标；
- `raw/`：B2–B5 逐次运行的原始实验 CSV；
- `summary/`：数据统计、供给侧冻结表、实验汇总、相关性和95%置信区间；
- `figures/`：B6 的 PDF、EPS、PNG 图；
- `figures/data/`：每张图对应的独立 CSV。

真实成都输入和生成结果已通过 `.gitignore` 排除，不应提交或重新分发；实验脚本、测试和冻结配置应纳入版本控制。

## 自然预测准确率 checkpoint 实验

该补充实验用于验证：预测模型通过正常训练逐步变准时，端到端调度性能是否随之变化。它与人工 prediction corruption 不同，研究的是训练过程中自然产生的模型参数和准确率梯度。

实验使用相同的 Histogram Gradient Boosting 模型，只改变训练轮数：

```text
max_iter = [1, 2, 5, 10, 20, 50, 100, 200]
```

以下内容对所有 checkpoint 保持完全一致：特征定义、数据切分、公交线路、容量、type value、Test 订单流、offline OPT、`theta=0.4` 和指标计算方法。

所有 checkpoint 只使用 Train 拟合。Validation 用于记录和排序模型准确率，Test 不用于选择 checkpoint 或调整超参数。滚动预测只能使用当前时点之前已经观测到的 lag，不读取未来需求。

### 独立运行 checkpoint 实验

```powershell
.\.venv\Scripts\python.exe scripts\train_checkpoint_predictions.py
.\.venv\Scripts\python.exe scripts\run_checkpoint_dispatch.py
.\.venv\Scripts\python.exe scripts\plot_checkpoint_experiment.py
```

每个 checkpoint 的预测会被显式转换为：

```text
{destination_station: predicted_count}
```

随后传入学生 A 的 predictive LP。同一个 checkpoint 和 slot 的 advice 只计算一次，并由 Prediction-only 与 RP-LAIPD 共享。IPD 不读取 prediction，因此作为应当保持不变的对照组。

默认调度实验使用12个均匀覆盖全天的 Test slot 和配置中的5个 seed。由于当前预测器和算法是确定性的，汇总时先对重复 seed 求均值，再将配对 Test slot 作为置信区间的有效统计单位，避免把完全相同的 seed 结果错误地视为独立样本。

### Checkpoint 实验结果

当前 Test 结果形成了清晰的自然准确率梯度：

| HGB max_iter | Test WAPE | Prediction-only ALG/OPT | RP-LAIPD ALG/OPT | IPD ALG/OPT |
|---:|---:|---:|---:|---:|
| 1 | 0.628 | 0.626 | 0.907 | 0.883 |
| 5 | 0.532 | 0.679 | 0.909 | 0.883 |
| 10 | 0.442 | 0.735 | 0.906 | 0.883 |
| 20 | 0.334 | 0.798 | 0.904 | 0.883 |
| 50 | 0.258 | 0.840 | 0.909 | 0.883 |
| 100 | 0.252 | 0.847 | 0.910 | 0.883 |
| 200 | 0.252 | 0.848 | 0.910 | 0.883 |

实验支持以下结论：

1. Prediction-only 随预测准确率提高而明显改善；
2. IPD 不依赖 prediction，对所有 checkpoint 完全不变；
3. RP-LAIPD 在低准确率 checkpoint 下仍保持稳定的 robust floor；
4. RP-LAIPD 在所有 checkpoint 上均高于 IPD；
5. demand WAPE 与 decision-aware advice error 不等价，必须同时报告。

跨8个 checkpoint 均值计算：

- Prediction-only 的 prediction error 与 ALG/OPT Pearson 相关为 `-0.999`；
- RP-LAIPD 的 prediction error 与 ALG/OPT Pearson 相关为 `-0.414`；
- IPD 对 checkpoint 完全不变，因此其 checkpoint 相关系数不定义。

### Checkpoint 输出文件

- `prediction/checkpoint_predictions.csv`：逐 checkpoint、slot、type 的预测；
- `prediction/checkpoint_prediction_metrics.csv`：Validation、Test、Shifted Test 指标；
- `summary/checkpoint_selection.csv`：只根据 Validation WAPE 生成的准确率排序；
- `raw/checkpoint_dispatch_results.csv`：逐 checkpoint、slot、seed、method 原始结果；
- `summary/checkpoint_dispatch_summary.csv`：以 slot 为单位的 mean、std、95% CI；
- `summary/checkpoint_correlations.csv`：误差与调度性能相关性；
- `figures/fig_checkpoint_*.pdf|eps|png`：三张可独立复现的 checkpoint 图；
- `figures/data/fig_checkpoint_*.csv`：每张 checkpoint 图对应的源数据。

## Distribution shift 与稳健性实验

当前实现四类 controlled prediction corruption：

1. Count scaling：`0.5x、0.75x、1.0x、1.25x、1.5x`；
2. Destination/type permutation：交换一定比例的 destination demand；
3. Temporal shift：将预测向前或向后移动若干 slot；
4. Scarce-resource adversarial concentration：将误差集中在紧缺资源和高价值 type。

运行：

```powershell
.\.venv\Scripts\python.exe scripts\run_distribution_shift.py
```

当前最强 destination permutation 下：

```text
IPD ALG/OPT ≈ 0.882
Prediction-only ALG/OPT ≈ 0.508
RP-LAIPD(theta=0.4) ALG/OPT ≈ 0.899
```

这说明 Prediction-only 会随着错误增大快速恶化，而 RP-LAIPD 保留了明显的 robust floor。

## Runtime 与 scalability

Runtime 实验分别改变：

```text
N = [100, 500, 1000, 5000]
M = [5, 10, 20, 50]
K = [10, 50, 100, 500]
```

记录 prediction time、predictive LP time、slot preparation time、slot runtime、average request latency 和 p95 request latency。

运行：

```powershell
.\.venv\Scripts\python.exe scripts\run_runtime_scalability.py
```

该实验用于验证 predictive LP 属于 slot-level overhead，而逐请求在线决策仍可保持较低延迟。

## 实验假设与限制

原论文指定了30个热门站点、随机生成的公交和订单、`(0,1]` 范围内的随机优先级、公交等待窗口以及四类资源约束，但没有公布精确站点坐标、随机生成器、CVRP 求解细节和全部数值常量。本实现将这些假设显式写入 YAML，并使用确定性坐标和固定 seed 保证可复现。

成都实验还存在以下限制：

- 成都数据是网约车订单，不是公交预约订单；
- 真实公交车辆、容量、线路和运营时间没有提供；
- 供给侧为使用 Train demand 冻结的半合成配置；
- 成都数据只有2016年11月一个月，不适合证明长期季节性和跨季节泛化；
- RP-LAIPD 当前属于可运行的 learning-augmented 启发式实现，不能将实验结果直接等同于完整理论保证；
- MAPE 在零需求 type 上容易失真，因此实验同时报告 WAPE、sMAPE 和 nonzero MAPE。

## 引用

```bibtex
@inproceedings{zhou2019realtime,
  title={Real-Time Route Planning and Online Order Dispatch for Bus-Booking Platforms},
  author={Zhou, Hao and Gao, Yucen and Gao, Xiaofeng and Chen, Guihai},
  booktitle={Database Systems for Advanced Applications},
  year={2019}
}
```
