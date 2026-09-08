# Learning-Augmented Online Order Assignment

> **Journal extension / student handoff status — 2026-09-09**
>
> 本 README 是当前项目的**第一入口**。实验数字与理论状态的权威快照见 [`docs/JOURNAL_STATUS.md`](docs/JOURNAL_STATUS.md)，论文级紧凑证据见 [`docs/evidence/`](docs/evidence/)。

## 0. 当前结论：不要再沿旧的 `theta` 路线继续调参

本项目已经从早期的 **RP-LAIPD（静态资源切分）** 收敛到冻结算法：

**VDR-LA with One-Sided Online Release**

当前判断是：

- 旧问题“IPD / 带预测算法还不如 Greedy、Random”不能靠继续调 `theta` 解决；
- RP-LAIPD 将资源永久切成 prediction branch 与 robust/IPD branch，强预测会被弱 strict-IPD 分支拖累，且两边不能共享剩余资源；
- 当前 VDR-LA 已在拥塞场景中体现出预测的实质价值，并保持 **strict feasibility**；
- 下一阶段的主要风险已经从“算法能否工作”转向：**strict-feasible IPD 理论、learning-augmented 理论闭环、更困难 benchmark、真实/公开轨迹与论文证据组织**。

因此，`theta` / RP-LAIPD 现在主要作为**历史路线和消融对照**保留，不再是 journal 主算法。

---

## 1. 研究问题

项目延续 DASFAA 2019 的在线订单分配问题：订单按时间在线到达，算法在不知道未来真实订单的情况下，将订单分配给可服务车辆，并最大化接受订单的总价值/优先级，同时满足真实资源约束。

当前实现中的核心资源约束包括：

1. bus capacity；
2. station capacity；
3. bus travel-time resource；
4. station travel-time resource。

主要评价指标：

```text
ALG / OPT = 在线算法获得的总价值 / 离线最优总价值
```

期刊扩展的核心问题不是“预测是否准确”本身，而是：

> **如何把不可靠的未来需求预测转化为在线容量保留决策，使好预测真正带来收益，同时在预测错误时不发生灾难性性能或可行性崩溃？**

---

## 2. 为什么原来的 IPD / RP-LAIPD 路线不够

### 2.1 DASFAA 2019 Paper-IPD 与 strict feasibility 不是同一个问题

会议版 IPD 的高竞争比与 **resource augmentation / constraint violation** 有关。因此 Paper-IPD 出现 `ALG/OPT > 1` 并不代表它在原始容量约束下超过离线最优，而是使用了放宽资源。

期刊版必须区分：

- **Paper-IPD**：复现 DASFAA 2019 legacy bicriteria result；
- **strict-feasible IPD**：所有真实容量约束严格满足，需要重新分析竞争性能。

不能把会议版 `(1-ε)` 型保证直接搬到 strict-feasible IPD。

### 2.2 RP-LAIPD 的结构性问题

`rp_laipd_dispatch` 使用静态资源切分：

- `theta = 0`：全部资源给 prediction-only；
- `theta = 1`：全部资源给 IPD；
- `0 < theta < 1`：按固定比例切分两套资源。

该设计的问题是：

1. **资源不可共享**：prediction branch 即使没有用完容量，也不能自然交给 robust branch；反之亦然；
2. **弱 backbone 拖累整体**：strict-IPD 自身在部分场景弱于 Greedy，切一部分资源给它会直接损失强预测分支的容量；
3. **固定 `theta` 不适应在线状态**：低负载、高负载、预测偏差和 arrival shift 对“应该保留多少容量”的要求不同；
4. **count error 不是全部**：真正影响决策的是“未来高价值需求是否被漏估/高估”，而非单一对称 WAPE。

因此主线从“切资源”改为“**共享资源上的预测引导保留 + 在线释放 + 安全可行决策**”。

---

## 3. 当前冻结技术路线：VDR-LA

正式算法：[`src/rood_dasfaa2019/algorithms/vdr.py`](src/rood_dasfaa2019/algorithms/vdr.py)

预测 LP 与历史 RP-LAIPD 代码：[`src/rood_dasfaa2019/algorithms/laipd.py`](src/rood_dasfaa2019/algorithms/laipd.py)

### 3.1 Ex-ante prediction → value-aware predictive advice

预测接口只允许使用 Train-side / 真正 ex-ante 信息：

- type-level demand count；
- type value / expected priority；
- arrival CDF。

这些信息进入 predictive LP，形成 `type × bus` quota。正式实验不读取 Test future。

### 3.2 Value-Dominance Reservation (VDR)

在线订单到达时：

1. 若当前 type 仍有 predictive quota，优先按 quota 接受；
2. 若不能直接按 quota 接受，则检查未来预测需求；
3. **只为价值严格高于当前订单的未来需求保留容量**；
4. 不再因为“未来有预测 quota”就机械拒绝当前高价值订单。

这是当前算法相对 prediction-only / static reservation 的核心机制。

### 3.3 One-Sided Online Release

预测错误具有方向性。当前算法只在有统计证据表明**实际累计到达显著低于预测轨迹**时，下调未来 reservation：

- 只释放被支持为 overprediction 的容量；
- 不因在线 calibration 主动把原预测放大；
- 若校准后的有效总需求已经不超过物理总容量，则进入 **work-conserving Greedy-like** 分支，不再继续无意义保留。

该机制主要修复 low-load overprediction、arrival shift 和 severe shifted count error。

### 3.4 Congestion-aware price + strict feasibility gate

在真正拥塞时，算法仍使用 primal-dual / congestion price 辅助筛选；但**任何 prediction、reservation、price 或 fallback 都不能绕过真实资源检查**。

每一次实际接受订单前都必须通过 strict feasibility gate，再执行 commit。

---

## 4. 当前最重要的实验结果

Official corrected **E8-B**：共 **14,400** 个 paired seed-level cases，且不再继续 calibration tuning。

| 指标 | 结果 |
|---|---:|
| Frozen VDR mean ALG/OPT | 0.9331 |
| Calibrated VDR mean ALG/OPT | **0.9383** |
| Mean gap vs Greedy | **+0.0668** |
| Perfect matched | **0.9703** |
| CDF shift | **0.9306** |
| Severe shifted count error | **0.9186** |
| Worst regime-mean gap vs Greedy | **-0.1214** |
| Strict-feasibility violations | **0** |

完整数字见 [`docs/evidence/E8B_REPORT.txt`](docs/evidence/E8B_REPORT.txt)。

### 如何理解 Greedy / Random 对比

必须按负载理解结果，而不是要求 VDR-LA 在每个 regime 都 pointwise 胜过 Greedy：

- load `0.8 / 1.0` 往往很容易，Random feasible scan 和 Greedy 都可能直接达到 OPT；此时复杂 reservation 没有多少收益空间；
- 高负载时稀缺容量才真正产生“应该为未来高价值订单保留资源”的问题；
- 在受控 bottleneck benchmark 的 load `2.0` 例子中，Random/Greedy 约为 **0.7125 ALG/OPT**，VDR-LA 约为 **0.8919**。

当前论文应该主张的是：**预测在拥塞、价值异质和未来竞争显著时提供决策价值，同时算法在错误预测下保持严格可行与可控退化**；不是“所有实例都严格优于 Greedy”。

---

## 5. 当前 learning-augmented 证据：误差是非对称的

受控 count-prediction 实验（CDF/value 保持正确）：

| count scale | mean ALG/OPT |
|---:|---:|
| 0.5× underprediction | 0.9072 |
| 1.0× perfect | **0.9703** |
| 1.5× overprediction | 0.9679 |
| 2.0× overprediction | 0.9524 |

另外：

- correct CDF vs shifted CDF：`0.9703 vs 0.9302`；
- exact value vs noisy value：`0.9703 vs 0.9660`；
- 在 load ≥ 1.2 的拥塞实例中，perfect prediction 比 0.5× underprediction 平均高约 **10.48 percentage points**。

关键结论：

> **漏估未来高价值需求（underprediction）通常比适度高估（overprediction）更危险。**

因此后续理论不再使用单一对称绝对误差，而是区分：

- `η-`：missed / underpredicted future high-value demand；
- `η+`：excess / overpredicted future high-value demand。

预期应有 `L- > L+`，即 underprediction 的性能惩罚更大。

---

## 6. 当前理论主线

### A. strict-feasible IPD competitive theory

保留 DASFAA 2019 bicriteria competitive guarantee 作为 legacy result；另外研究 strict-feasible IPD 在原始容量约束下的竞争保证。

这一步非常重要，因为 strict-feasible IPD 还要作为 learning-augmented 算法的 robust backbone / safety envelope。

### B. Learning-augmented consistency / robustness / smoothness

目标是标准三件套：

1. **Consistency**：预测准确时接近 predictive optimum / OPT（在明确假设下）；
2. **Robustness**：任意预测误差下，由 strict-feasible IPD / safety backbone 提供全局下界；
3. **Asymmetric smoothness**：性能随 `η-`、`η+` 退化，而不是随一个对称误差线性退化。

当前目标形式可写为：

```text
ALG >= max{ rho_robust,
           1 - delta - L_- * eta_- - L_+ * eta_+ } * OPT
```

**注意：这是目标 theorem form，目前不是已经证明的最终定理。**

---

## 7. 下一阶段工作：学生应优先做什么

### Track T — 理论

优先级 P0：

1. 推导 strict-feasible IPD competitive guarantee；
2. 明确 lower bound / tightness，确认能否作为 VDR-LA 的 global robustness backbone；
3. 形式化 VDR-LA consistency；
4. 用 `η- / η+` 建立 asymmetric smoothness，或证明一段 overprediction tolerance 区间；
5. 明确哪些结论需要 unit-demand / seat-bottleneck 等假设，避免过度声称。

### Track E — 实验

优先级 P0/P1：

1. 建立**更困难 synthetic benchmark**：非等价 bus/route/candidate set，重点看 load ≥ 1.2；
2. 完成 prediction error × load × value heterogeneity × arrival shift 的主表与 95% CI；
3. 形成 clean ablation：
   - count-only → value-aware advice；
   - no reservation → VDR；
   - fixed forecast → one-sided release；
   - static partition → shared reservation/fallback；
4. 加入一个真实/公开 mobility trace；成都数据可用则补充，但不作为唯一阻塞项；
5. 专门保留 failure-regime analysis，解释哪些场景 Greedy 仍更强以及原因。

### 暂时不要做

- 不再继续大规模扫描 `theta`；
- 不再为了逐 regime 压过 Greedy 增加新的 calibration threshold；
- 不要用 Test future 构造 prediction advice；
- 不要把 Paper-IPD 的 `ALG/OPT > 1` 与 strict-feasible 算法直接比较并解释为“更优”；
- 不要再使用单一对称 WAPE 作为 smoothness theorem 的唯一误差变量。

---

## 8. 代码与证据阅读顺序

学生第一次接手建议按下面顺序阅读：

1. **本 README**：理解路线转折和当前任务；
2. [`docs/JOURNAL_STATUS.md`](docs/JOURNAL_STATUS.md)：当前权威技术/实验快照；
3. [`src/rood_dasfaa2019/algorithms/vdr.py`](src/rood_dasfaa2019/algorithms/vdr.py)：当前冻结算法；
4. [`src/rood_dasfaa2019/algorithms/laipd.py`](src/rood_dasfaa2019/algorithms/laipd.py)：predictive LP + 历史 RP-LAIPD，理解为什么旧路线失败；
5. [`src/rood_dasfaa2019/algorithms/ipd.py`](src/rood_dasfaa2019/algorithms/ipd.py)：IPD 实现；
6. [`docs/evidence/E8B_REPORT.txt`](docs/evidence/E8B_REPORT.txt)：E8-B 关键数字；
7. [`docs/evidence/smoothness/`](docs/evidence/smoothness/)：方向性误差证据；
8. [`tests/test_vdr.py`](tests/test_vdr.py) 与 [`tests/test_algorithms.py`](tests/test_algorithms.py)：算法一致性与基本回归测试。

---

## 9. 安装与基本检查

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .
pytest -q
```

历史 DASFAA 2019 复现实验仍可通过：

```bash
python scripts/reproduce_paper.py --seed 2019
```

方向性 smoothness 分析入口：

```bash
python scripts/analyze_directional_smoothness.py
```

> 大规模 E8 原始网格与 Didi/Chengdu processed data 不在 Git 中跟踪；仓库只保留紧凑、可用于论文复核的证据文件。

---

## 10. Provenance

本仓库是基于论文的 reimplementation / journal-extension codebase。原 DASFAA 2019 历史源码不可用，因此会议版实现应描述为 **paper-based reimplementation**，而不是原始源码的精确恢复。

Conference paper:

> *Real-Time Route Planning and Online Order Dispatch for Bus-Booking Platforms*, DASFAA 2019.

仓库中保留 [`DASFAA_2019.pdf`](DASFAA_2019.pdf) 作为原论文参考。
