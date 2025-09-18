# Web3CausalInference

面向 **Web3 运营活动评估** 的因果推断工具箱：在链上/钱包业务数据上，稳健地估计 **ATE/ATT/HTE**，并用 **nAUUC** 进行人群排序与投放分层优化。核心组件包含：

* **DR-Learner / ForestDRLearner**：异质处理效应（HTE）与人群排序
* **AIPW / TMLE 基线估计**：ATE/ATT 点估计与置信区间
* **nAUUC 调参器**：以 **nAUUC 最大化**为主、兼顾估计稳定性（可配置）
* **重叠与权重诊断**：OVL / KS / ESS / 权重尾部 / 安慰剂与负对照
* **Web3 业务友好**：适配链上活动、分地区/分层投放、长期效应建模（T+30 等）

> 本仓库使用 **MIT License**。([GitHub][1])

---

## ✨ 特性

* **双重稳健 ATE/ATT**：AIPW/TMLE 在倾向得分或结局模型任一正确时仍一致
* **nAUUC-first**：以 uplift 排序为主目标，提升运营分层投放回报
* **交叉拟合**：OOF 预测减少过拟合与信息泄露
* **诊断一体化**：重叠性、平衡性、影响函数尾部、安慰剂置换等一键输出
* **CatBoost 友好**：原生处理缺失值、类别特征；支持 Tweedie 回归用于金额型目标

---

## 🔧 安装

```bash
# Python 3.9+ 建议
pip install -U numpy pandas scikit-learn catboost econml tqdm
```

> 提示：macOS 不支持 CatBoost 的 CUDA GPU；建议本地 CPU 调试、服务器上用 NVIDIA GPU 训练。仓库许可为 MIT。([GitHub][1])

---

## 🚀 快速上手

下面演示典型评估流：**用 nAUUC 调参 → 训练 DR-Learner 拿 HTE → 计算 ATE/ATT 与诊断报表**。

```python
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier
from econml.dr import ForestDRLearner

# 你的数据：X(协变量), T(是否参与活动/是否戒烟), Y(效果，如 wt82-wt71 或 30天交易额增量)
X: pd.DataFrame = ...
T: np.ndarray   = ...
Y: np.ndarray   = ...

# 1) 用 nAUUC 调参，返回“未拟合”的最佳 reg / clf（与你现有 API 对齐）
from src.nauuc_tuner import NAUUCCatBoostTuner  # 假设你的类路径为此
tuner = NAUUCCatBoostTuner(n_trials=60, n_splits=5, random_seed=42, verbose=True)
best_reg, best_clf = tuner.fit_return_models(X, T, Y)

# 2) 训练 DR-Learner，得到 HTE / uplift 排序
dr = ForestDRLearner(model_regression=best_reg,
                     model_propensity=best_clf,
                     random_state=42, categories='auto')
dr.fit(Y, T, X=X)
tau_hat = dr.effect(X)          # 每个用户的个体效应，用于排序/圈人
lb, ub  = dr.effect_interval(X) # 可选：HTE 区间

# 3) 估计 ATE/ATT（简洁示例）
#    推荐同时输出诊断指标：OVL/KS/ESS、权重尾部、SMD、安慰剂等（见下文 Best Practices）
ate = tau_hat.mean()
print(f"ATE ≈ {ate:.3f}")
```

> 如果你在跑 **NHEFS** 数据：默认口径是 `T=qsmk（1=戒烟）`、`Y=wt82-wt71`，合理 ATE 在 **+3 \~ +4** 左右；切勿对数据随意 `dropna`，否则会引入选择偏差。参考仓库当前结构（`README.md`、`src/`、`example/`）。([GitHub][2])

---

## 📈 nAUUC 调参与使用建议

* **目标**：最大化验证集 **nAUUC**（以 AIPW 伪效应作“oracle”基准）
* **可选副目标**：在不改 API 的前提下，用验证折的 **AIPW 影响函数 φ 估计的 SE** 来惩罚 CI 宽度（`objective = nAUUC - λ·CIw`）
* **排序/投放**：根据 `tau_hat` 由高到低圈人，绘制 uplift 曲线（Top q% 覆盖率）

---

## ✅ 诊断与最佳实践

为了把 **CI 收紧**、防止错号或“看起来稳定实则偏”的情况，建议在训练/评估阶段开启以下最小诊断（不影响对外 API）：

1. **缺失值**

   * 不要对训练样本 `dropna`。数值 NaN 直接交给 CatBoost；分类缺失单列成“一类”。
   * 如果 **Y** 有缺失，用 **IPCW**（建模 `P(R=1|X)`，在 φ/SE 里乘 `R/π(X)`）代替删样本。

2. **交叉拟合**

   * 倾向与结局模型均做 **K 折 OOF**；避免泄漏。

3. **重叠性**

   * 固定裁剪：`e(x) ∈ [0.05, 0.95]`；报告 **OVL/KS/ESS** 与 **权重尾部 (p99/max)**。
   * 若重叠差，用 trimming / re-weighting 或改特征抽样窗口。

4. **CUPED 残差化（可选）**

   * 先拟合 `g(X)≈E[Y|X]`，用 `Y* = Y - ĝ(X)` 训练/评估，常能显著缩小 CI。

5. **Sanity Check（以 NHEFS 为例）**

   * 朴素差值 `mean(Y|T=1)-mean(Y|T=0)` 应为 **正**；
   * AIPW/DR/TMLE 的 ATE 同号且量级接近常识（NHEFS ≈ +3\~4）。

---

## 🧭 目录结构（示例）

```
Web3CausalInference/
├─ src/
│  ├─ nauuc_tuner.py        # nAUUC 调参器（返回 CatBoost reg/clf）
│  ├─ estimators.py         # AIPW / TMLE / DR-Learner 包装（可选）
│  ├─ diagnostics.py        # OVL/KS/ESS、SMD、权重尾部、φ 等诊断
│  └─ utils.py              # cross-fitting、CUPED、IPCW 等工具
├─ example/
│  ├─ nhefs_demo.ipynb      # NHEFS 示例（戒烟→体重变化）
│  └─ web3_campaign.ipynb   # Web3 活动评估示例（如 T+30 swap 增量）
├─ README.md
└─ LICENSE                  # MIT
```

> 实际文件名以你仓库为准；此处给出建议性组织方式，便于后续扩展与他人协作。仓库当前标注 **MIT license**。([GitHub][1])

---

## 🧪 典型用例

* **Web3 活动因果评估**：对“长期余额/交易额增量（T+30）”做 AIPW/DR；nAUUC 圈选高响应用户，支持“分地区/分层投放”
* **重叠性较差**：启用裁剪 + 诊断 + CUPED，必要时在 `X` 中加入 pre-trend 或更稳健的基础协变量
* **排查异常**：ATE 错号或超宽 CI 常见于 **dropna 选择偏差**、`T/Y` 编码错误、或 e(x) 极端未裁剪

---

## 🗺️ 路线图（建议）

* `report()`：一键输出 ATE/ATT 诊断（OVL/KS/ESS、SMD、权重尾部、φ 尾部、安慰剂/负对照、uplift 曲线）
* `tmle.py`：加入 TMLE 复核
* `metrics.py`：nQini/AUUC/nAUUC 的统一实现与可视化
* `datasets/`：内置 NHEFS & 合成数据便于复现实验

---

## 🤝 贡献

欢迎 Issue / PR：补充 Web3 业务数据清洗范式（链上价格校正、事件时间展开、地域映射等），以及更强的诊断与可视化。

---

## 📜 许可证

本项目基于 **MIT License** 开源。详见 [`LICENSE`](./LICENSE)。([GitHub][1])
