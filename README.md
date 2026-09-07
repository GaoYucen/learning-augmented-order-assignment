# 成都订单预测增强调度实验（学生 B）

本分支 `yzj-prediction-experiment` 在学生 A 的调度算法接口上，重新实现了无未来信息泄漏的需求预测实验，以及固定预测器下的 θ—误差敏感性实验。正式结果使用成都订单构造的“抽样实验需求”：每个 30 分钟 service slot 最多抽样 300 单，每单按 1 人处理。

## 统一定义

- request type 是目的站点，预测标签、站点真实需求、predictive LP 和在线调度使用同一口径。
- Train 只用于拟合模型和数据变换；Validation 只用于超参数、损失函数、融合权重和迭代数选择；Test 只用于最终评价。
- 站点映射、订单价值、线路、车辆和资源配置均不使用未来 Test 需求。
- θ 是分给稳健 IPD 分支的资源比例：`θ=0` 等价于 Prediction-only，`θ=1` 等价于 IPD；θ 越大表示越保守。
- `ALG/OPT` 是在线算法收益与已证明最优的离线收益之比，越大越好。未证明最优的时段不会混入严格 OPT 比较。

## 预测器

保留三个模型，且不预设哪个模型一定最好：

- `HistoricalAverage`：在“站点×星期×时段”和“站点×时段”历史均值间融合，融合权重按 Validation MAE 选择；原始细分历史均值另行保存比较。
- `RidgeAR`：目标站点固定 one-hot，`StandardScaler + Ridge` 组成 Pipeline；Scaler 仅从 Train 拟合，`alpha=[0.01,0.1,1,10,100]` 按 Validation MAE 选择。
- `HGB`：站点编号作为类别特征，关闭自动早停；平方误差/Poisson 损失与最大迭代数均按独立时间 Validation 集选择，并保存实际迭代数。

所有输出预测非负。对目标时段 `t` 和跨度 `h`，预测起点是 `t-h`；需求特征仅能读取预测起点及其之前已经结束的时段，目标时段的日历信息可以使用。每次预测的目标始终是一个 30 分钟时段，而不是多个时段的累计需求。

## 两张最终图

### 图一：预测跨度与预测误差

`fig1_forecast_horizon_mae.png/pdf`

- 横轴：`h=[1,2,4,8]`，即提前 30、60、120、240 分钟预测。
- 纵轴：共同 Test“时段×站点”观测上的 MAE，越低越好。
- 三条曲线分别为 HistoricalAverage、RidgeAR、HGB，误差条为按完整 service date 分块重采样得到的 95% 置信区间。
- 曲线完全来自真实模型输出，允许平坦、交叉或非单调，不人为缩放预测值。

同时保存 MSE、RMSE、WAPE、非零需求 MAPE，以及 MAPE 排除的零需求比例。

### 图二：站点预测与 θ—误差敏感性

`fig2_station_prediction_theta.png/pdf`

- 左侧使用 `h=1`。从共同 Test 时段中选择总需求最接近中位数的代表时段；若并列，按时间从早到晚选择。柱形是真实站点需求，三条曲线是未经扰动的三个模型预测。
- 右侧只用 Validation 的 `h=1` MAE 选出并固定一个预测器，再加入站点尺度归一化高斯噪声。横轴是扰动后实际测得的 MAE，不是噪声系数 λ；纵轴是 ALG/OPT；六条曲线对应 `θ=[0,0.2,0.4,0.6,0.8,1]`。
- 使用 10 个固定扰动种子；同一种子在不同 λ 下复用相同噪声方向，同一扰动预测供所有 θ 使用。置信区间保留种子和 service date 的配对关系。

图二右侧只说明所设高斯扰动条件下的经验敏感性，不能替代理论最坏情况保证。

## 运行

在仓库根目录安装依赖并执行完整流程：

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts\run_chengdu_pipeline.py
```

若 `outputs/chengdu/processed/` 已存在，可跳过数据预处理：

```powershell
.\.venv\Scripts\python.exe scripts\run_chengdu_pipeline.py --skip-data
```

只重新训练预测器、运行 θ 实验并生成结果：

```powershell
.\.venv\Scripts\python.exe scripts\run_final_prediction_experiments.py
```

仅读取已保存 CSV 重绘两张图：

```powershell
.\.venv\Scripts\python.exe scripts\plot_final_prediction_figures.py --input-dir outputs/chengdu/final_experiment
```

运行测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

默认配置为 `configs/chengdu.yaml`，原始订单目录为 `D:/route规划/chengdu order`。

## 输出文件

所有正式产物保存在 `outputs/chengdu/final_experiment/`：

- `forecast_predictions.csv`：逐模型、跨度、预测起点、目标时段、站点的真实值和预测值；
- `forecast_metrics.csv`：预测指标和 MAE 置信区间；
- `historical_original_predictions.csv`、`historical_original_metrics.csv`：原始历史均值对照；
- `validation_model_selection.csv`、`theta_predictor_selection.csv`：仅依据 Validation 的选择记录；
- `models/`、`training_manifest.csv`、`feature_definition.json`：模型、训练范围、特征和选择依据；
- `representative_slot_selection.csv`、`representative_station_predictions.csv`：代表时段选择与图二左侧源数据；
- `perturbed_predictions.csv`、`perturbation_settings.csv`：逐站点扰动预测和固定噪声设置；
- `theta_dispatch_results.csv`、`theta_error_summary.csv`：逐时段调度结果与图二右侧汇总；
- `opt_status.csv`：离线求解状态、gap 和严格最优标记；
- `theta_endpoint_validation.csv`：θ 两端的分配与收益一致性检查；
- `resolved_config.yaml`、`dependency_versions.csv`、`experiment_manifest.json`：可复现实验配置和环境信息。

主要入口是 `scripts/run_final_prediction_experiments.py`，绘图入口是 `scripts/plot_final_prediction_figures.py`，预测实现位于 `src/rood_dasfaa2019/learning/real_prediction.py`。
