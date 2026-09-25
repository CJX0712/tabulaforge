# TabulaForge

> 模块化自动化表格机器学习（AutoML Tabular）—— 复用顶级开源（XGBoost · LightGBM · Optuna · scikit-learn · pandas），零下载可跑，一键复现。
> 作者：**晨星**

TabulaForge 是一套**生产可用**的表格机器学习系统：以 [XGBoost](https://github.com/dmlc/xgboost) 与 [LightGBM](https://github.com/microsoft/LightGBM) 为核心梯度提升引擎（20+ 超参可调），[Optuna](https://github.com/optuna/optuna) 负责超参搜索，scikit-learn 负责标准化与指标，numpy/pandas 提供零依赖兜底。模块按单一职责划分，接口契约先行，每个模块可独立验证。

## 特性

- ✅ **复用顶级开源**：XGBoost / LightGBM（业界 SOTA 梯度提升）+ Optuna（顶级 HPO）+ scikit-learn（指标/兜底）。
- ✅ **离线可跑兜底**：XGBoost / LightGBM / Optuna 缺失时自动降级纯 scikit-learn + numpy 随机搜索，**零下载即可运行 demo**。
- ✅ **模块化 + 契约先行**：`core` 定义类型/错误码/配置/Protocol，`data → hpo → training → models → eval → pipeline` 单向无环。
- ✅ **分类 + 回归统一**：同一流水线覆盖两类任务，指标自动切换。
- ✅ **超参搜索**：Optuna TPE 优先，缺失时 numpy 随机搜索兜底。
- ✅ **一键复现**：`requirements.lock.txt` 锁定全部依赖，Makefile / Dockerfile 即开即用。
- ✅ **量化基线**：内置 benchmark 横向评测（ROC-AUC / Acc / F1 / R2 / RMSE / MAE）。

## 快速开始

```bash
# 1. 创建虚拟环境并安装依赖
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 2. 运行端到端 demo（合成分类 + 回归 + HPO 评测）
python -m tabulaforge.examples.run_demo

# 3. 运行单测
pytest -q

# 4. 命令行：单模型
python -m tabulaforge.cli --task classification --model xgboost

# 5. 命令行：横向 benchmark（含 HPO）
python -m tabulaforge.cli --task classification --benchmark --hpo
```

## 代码结构

```
tabulaforge/
├── core/            # 契约层：types / errors / config / interfaces
├── data/            # 合成数据生成 + CSV/parquet 载入
├── models/          # XGBoost/LightGBM 封装 + sklearn 兜底 + factory
├── hpo/             # Optuna 超参搜索（numpy 随机搜索兜底）
├── training/        # 拟合编排
├── eval/            # 分类/回归评测指标
├── pipeline/        # TabulaPipeline + benchmark
├── cli.py           # argparse 入口
└── examples/        # run_demo + benchmark.json
tests/               # pytest 单测
docs/architecture.md # 架构设计
```

## 作为库调用

```python
from tabulaforge import Config, TabulaPipeline
from tabulaforge.data.loaders import make_classification

cfg = Config(task="classification", model="xgboost", use_hpo=True, n_trials=30)
pipe = TabulaPipeline(cfg)
metrics = pipe.run()          # 自动生成合成数据并评测
print(metrics.as_dict())
```

## 模型一览

| 名称 | 后端 | 需额外依赖 |
|------|------|-----------|
| xgboost | XGBoost 3.4.1 | xgboost |
| lightgbm | LightGBM 4.7.0 | lightgbm |
| sklearn | scikit-learn | —（零下载）|

## 评测基线

数据集 3000 样本 / 20 特征，seed=42：

| Task | Model | 主指标 |
|------|-------|--------|
| 分类 | xgboost | ROC-AUC 0.9189 / Acc 0.8856 |
| 分类(HPO) | xgboost | ROC-AUC 0.9202 / Acc 0.8911 |
| 回归 | lightgbm | R2 0.9462 / RMSE 45.69 |
| 回归 | sklearn | R2 0.9435 / RMSE 46.80 |

完整基线见 `docs/architecture.md` 与 `tabulaforge/examples/benchmark.json`。

## 验收（DoD）

- [x] 干净环境 clone → 一键 demo 零手工干预
- [x] 模块单测全通过（28 passed）
- [x] 依赖锁定可复现（`requirements.lock.txt`）
- [x] 文档覆盖架构 / 部署 / 使用
- [x] 关键指标量化基线 + 对照

## 许可证

MIT © 晨星
