"""TabulaForge 端到端演示。

零下载可跑：生成合成分类与回归数据，跨 XGBoost / LightGBM / scikit-learn
（含可选 HPO）做横向评测，打印对比表并落盘 benchmark.json。
作者：晨星
"""

from __future__ import annotations

import json
import os

from ..core.config import Config
from ..data.loaders import make_classification, make_regression
from ..models.factory import available_models
from ..pipeline.tabula_pipeline import benchmark


def _run(task: str, use_hpo: bool) -> list:
    cfg = Config(
        task=task,
        n_samples=3000,
        n_features=20,
        n_classes=2,
        noise=0.1,
        random_state=42,
        n_trials=25,
        use_hpo=use_hpo,
    )
    if task == "classification":
        ds = make_classification(
            n_samples=cfg.n_samples, n_features=cfg.n_features,
            n_classes=cfg.n_classes, noise=cfg.noise, random_state=cfg.random_state,
        )
    else:
        ds = make_regression(
            n_samples=cfg.n_samples, n_features=cfg.n_features,
            noise=cfg.noise, random_state=cfg.random_state,
        )
    names = available_models()
    return benchmark(names, config=cfg, dataset=ds, use_hpo=use_hpo)


def _print(task: str, metrics: list) -> None:
    print(f"\n=== {task.upper()} 评测" + (" (HPO)" if metrics and metrics[0].hpo else "") + " ===")
    if task == "classification":
        print(" | ".join(f"{h:<10}" for h in ["Model", "ROC-AUC", "Acc", "F1", "HPO"]))
        print("-" * 52)
        for m in metrics:
            mt = m.metrics
            print(" | ".join([
                f"{m.name:<10}",
                f"{mt.get('roc_auc', float('nan')):<10.4f}",
                f"{mt.get('accuracy', float('nan')):<10.4f}",
                f"{mt.get('f1_macro', float('nan')):<10.4f}",
                f"{'Y' if m.hpo else '-':<10}",
            ]))
    else:
        print(" | ".join(f"{h:<10}" for h in ["Model", "R2", "RMSE", "MAE", "HPO"]))
        print("-" * 52)
        for m in metrics:
            mt = m.metrics
            print(" | ".join([
                f"{m.name:<10}",
                f"{mt.get('r2', float('nan')):<10.4f}",
                f"{mt.get('rmse', float('nan')):<10.4f}",
                f"{mt.get('mae', float('nan')):<10.4f}",
                f"{'Y' if m.hpo else '-':<10}",
            ]))


def run_demo() -> None:
    print(f"[demo] 可用模型: {available_models()}")
    cls = _run("classification", use_hpo=False)
    _print("classification", cls)
    reg = _run("regression", use_hpo=False)
    _print("regression", reg)
    cls_hpo = _run("classification", use_hpo=True)
    _print("classification", cls_hpo)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "classification": [m.as_dict() for m in cls],
                "regression": [m.as_dict() for m in reg],
                "classification_hpo": [m.as_dict() for m in cls_hpo],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n[demo] 基线已写入: {out_path}")


if __name__ == "__main__":
    run_demo()
