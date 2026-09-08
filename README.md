<!-- JOURNAL-STATUS-2026-09-09 -->
# Learning-Augmented Online Order Assignment

**Current journal status (2026-09-09):** the experimental algorithm is frozen as **VDR-LA with One-Sided Online Release**. The official corrected E8-B has 14,400 paired cases, mean ALG/OPT 0.9383, perfect matched 0.9703, and zero strict-feasibility violations. Formal `vdr_la_dispatch` is aligned with the official experiment implementation.

The current theory program has two parts: (1) re-analyze IPD under strict feasibility rather than silently inheriting the conference resource-augmentation guarantee; (2) establish learning-augmented consistency, global robustness via a strict-IPD backbone, and **asymmetric** smoothness using directional future-high-value-demand errors.

See [`docs/JOURNAL_STATUS.md`](docs/JOURNAL_STATUS.md) for the authoritative technical/evidence snapshot and `docs/evidence/` for compact paper-relevant summaries and figures. Large raw outputs and Didi/Chengdu processed data are intentionally not tracked.

---
# 成都订单预测增强调度实验（学生 B）

本分支在学生 A 的 `hyk-core-algorithms` 最新实现上，接入成都真实订单预测，并只保留两类最终图，用于回答两个问题：

1. 不同预测模型、不同预测准确率，是否对应不同的调度表现；
2. 各站点预测与真实需求有何差异，以及预测误差变化时，不同 `theta` 的表现如何。

## theta 的统一定义

当前实现采用资源划分（resource partition）定义：

- `theta = 0`：100% 资源用于预测建议，完全相信预测，等价于 `Prediction-only`；
- `theta = 1`：100% 资源用于稳健分支，完全不使用预测，等价于 `IPD`；
- `0 < theta < 1`：`theta` 比例的资源给 IPD，`1-theta` 比例的资源给预测建议。

因此，`theta` 越大表示越保守、越不相信预测，而不是越相信预测。

## 最终实验设计

实验在同一批均匀抽取的 Test 时段上比较：

- `HistoricalAverage`：使用 `0.40、0.55、0.70、0.85、1.00` 五个预测校准比例形成准确率梯度；
- `RidgeAR`：使用相同的五个预测校准比例形成准确率梯度；
- `HGB`：使用 `1、2、5、10、20、50、100、200` 次训练迭代形成自然准确率梯度。

预测指标使用 Test WAPE：

```text
WAPE = sum(|预测订单数 - 真实订单数|) / sum(真实订单数)
```

WAPE 越小，预测越准确。调度指标使用：

```text
ALG / OPT = 在线算法接受订单的总价值 / 离线最优总价值
```

ALG/OPT 越接近 1，在线算法越接近知道全部订单后得到的离线最优结果。

所有模型使用相同 Test 时段、车辆、路线、容量和离线 OPT，保证横向比较公平。预测模型与 request type 的价值只用 Train 数据训练或估计，不读取 Test 时段的未来真实需求。

## 运行方法

在仓库根目录执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts\run_chengdu_pipeline.py
```

如果成都数据和预测 CSV 已经生成，可直接复用：

```powershell
.\.venv\Scripts\python.exe scripts\run_chengdu_pipeline.py --skip-data --skip-prediction
```

也可以只重新运行最终实验和绘图：

```powershell
.\.venv\Scripts\python.exe scripts\run_final_prediction_theta_figures.py
```

默认从 `configs/chengdu.yaml` 读取配置，原始成都订单目录为 `D:/route规划/chengdu order`。

## 最终输出

结果位于 `outputs/chengdu/final_figures/`。最终只生成两类图：

1. `figure_1_model_accuracy_dispatch.png/pdf`
   - 横轴：模型版本或训练阶段；
   - 纵轴：预测准确性 `1 - Test WAPE`，越高越准确；
   - Historical Average、RidgeAR、HGB 各自形成一条预测模型曲线；
   - 每个点代表该模型的一个准确率版本，本图只评价预测器本身，不再混入调度指标。

2. `figure_2_station_prediction_and_theta.png/pdf`
   - 左图：代表性 Test 时段中，各目的站点的真实订单数与多种模型预测值；
   - 右图：横轴为 Test WAPE、纵轴为 ALG/OPT，曲线分别对应 `theta=0、0.2、0.4、0.6、0.8、1`。

同时保存可复核的源数据：

- `final_model_theta_raw.csv`：逐时段原始结果；
- `final_model_theta_summary.csv`：按模型、算法与 theta 汇总；
- `station_prediction_source.csv`：站点真实值和预测值；
- `theta_endpoint_validation.csv`：验证 `theta=0` 等价于 Prediction-only、`theta=1` 等价于 IPD。

## 核心代码

- `src/rood_dasfaa2019/algorithms/laipd.py`：学生 A 的资源划分 RP-LAIPD，并保留学生 B 的外部预测与 Train-only type value 接口；
- `src/rood_dasfaa2019/experiments/chengdu.py`：成都时段、车辆路线、预测 advice、OPT 与在线算法的统一实验入口；
- `scripts/train_chengdu_predictions.py`：训练基础预测模型；
- `scripts/train_checkpoint_predictions.py`：生成不同训练迭代次数的 HGB 模型结果；
- `scripts/run_final_prediction_theta_figures.py`：重新运行配对实验并生成两类最终图。
