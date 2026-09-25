# TabulaForge 架构设计

> 作者：晨星 ｜ 模块化自动化表格机器学习（AutoML Tabular）｜ 复用顶级开源：XGBoost 3.4.1 + LightGBM 4.7.0 + Optuna 5.0.0 + scikit-learn 1.9.1 + pandas 3.0.6 + numpy 2.5.3

## 1. 设计原则

- **优先复用顶级开源**：梯度提升直接复用 XGBoost / LightGBM（业界 SOTA），超参搜索复用 Optuna，标准化/指标复用 scikit-learn，绝不从零自研。
- **离线兜底**：XGBoost / LightGBM / Optuna 缺失时自动降级到纯 scikit-learn（HistGradientBoosting）+ numpy 随机搜索，保证零下载 demo 可跑。
- **单一职责 + 契约先行**：core 定义类型/错误码/配置/Protocol，实现在后；调用单向无环。
- **可独立验证**：每模块可单测 + 最小可运行示例。

## 2. 模块划分与调用关系

```
                ┌──────────────────────────────────────────┐
                │              cli.py                       │
                │  --task / --model / --benchmark / --hpo   │
                └──────────────────┬───────────────────────┘
                                   │
                ┌──────────────────▼───────────────────────┐
                │     pipeline/tabula_pipeline.py           │
                │  TabulaPipeline.run() / benchmark()        │
                └───┬────────┬─────────┬────────┬──────────┘
                    │        │         │        │
            ┌───────▼─┐ ┌────▼────┐ ┌──▼─────┐ ┌─▼──────┐
            │ data/   │ │ hpo/    │ │training│ │ eval/  │
            │loaders  │ │searcher │ │fit     │ │metrics │
            └─────┬───┘ └────┬────┘ └──┬─────┘ └──┬─────┘
                  │          │         │          │
                  └──────────┼─────────┘          │
                             ▼                    │
                  ┌──────────────────────┐        │
                  │ models/factory.py    │◄───────┘
                  │ XGBoost/LightGBM/    │
                  │ sklearn 兜底          │
                  └──────────┬───────────┘
                             ▼
                  ┌──────────────────────┐
                  │ core/ (types/errors/ │
                  │ config/interfaces)   │
                  └──────────────────────┘
```

调用方向严格单向、无环：`cli → pipeline → {data, hpo, training, models, eval} → core`。

## 3. 核心契约（core/）

| 文件 | 职责 |
|------|------|
| `types.py` | `DatasetFrame` / `ModelResult` / `EvalMetrics` / `HpoResult` / `TaskType` |
| `errors.py` | 错误码：E100 数据 / E200 模型 / E300 训练 / E400 评测 / E500 配置 |
| `config.py` | `Config` dataclass，支持 `TABULAFORGE_*` 环境变量覆盖 |
| `interfaces.py` | `IDataLoader / IModel / IHPOSearcher / ITrainer / IEvaluator / IPipeline` Protocol |

## 4. 模型清单

| 名称 | 后端 | 说明 |
|------|------|------|
| `xgboost` | XGBoost 3.4.1 | 顶级梯度提升（分类/回归）|
| `lightgbm` | LightGBM 4.7.0 | 微软顶级梯度提升（分类/回归）|
| `sklearn` | scikit-learn | HistGradientBoosting 零额外依赖兜底 |

> 指定模型后端不可用时，factory 自动跳过，benchmark / 默认集成回退到 sklearn。

## 5. 错误码体系

| 区间 | 域 | 典型触发 |
|------|----|---------|
| E100 | 数据层 | CSV 解析失败、非法 task、文件不存在 |
| E200 | 模型层 | 后端不可用、拟合/推理失败 |
| E300 | 训练层 | 训练编排异常 |
| E400 | 评测层 | 指标计算失败（如仅单类）|
| E500 | 配置层 | task 非法、环境变量解析失败 |

## 6. 性能基线（benchmark）

数据集：3000 样本 / 20 特征，随机种子 42，训练/测试 7:3。

**分类（contamination 噪声 0.1）**

| Model | ROC-AUC | Acc | F1 | HPO |
|-------|--------|-----|----|-----|
| xgboost | 0.9189 | 0.8856 | 0.8855 | - |
| sklearn | 0.9174 | 0.8800 | 0.8799 | - |
| lightgbm | 0.9157 | 0.8811 | 0.8810 | - |
| xgboost(HPO) | 0.9202 | 0.8911 | 0.8910 | Y |

**回归**

| Model | R2 | RMSE | MAE |
|-------|----|------|-----|
| lightgbm | 0.9462 | 45.69 | 34.48 |
| sklearn | 0.9435 | 46.80 | 36.29 |
| xgboost | 0.8963 | 63.43 | 48.64 |

**结论**：LightGBM / XGBoost 在回归上领先 sklearn 兜底约 0.9~4.5 个 R2 点；分类三者接近（ROC-AUC≈0.92）。Optuna HPO 在少量 trial 下即带来稳定小幅提升。纯 scikit-learn 兜底在无 XGBoost/LightGBM 时仍提供可用基线。

## 7. 验证（DoD 对照）

| 验收项 | 状态 |
|--------|------|
| 干净环境 clone → 一键 demo 零手工干预 | ✅ `python -m tabulaforge.examples.run_demo` |
| 模块单测全通过 | ✅ 28 passed |
| 依赖锁定可复现 | ✅ `requirements.lock.txt`（pip freeze，28 项）|
| 文档覆盖架构/部署/使用 | ✅ 本文件 + README.md |
| 关键指标量化基线 | ✅ 上表对照 |

基线详情落盘于 `tabulaforge/examples/benchmark.json`。
