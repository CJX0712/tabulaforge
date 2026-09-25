"""TabulaForge 命令行入口。作者：晨星

示例：
  python -m tabulaforge.cli --task classification --model xgboost
  python -m tabulaforge.cli --task classification --benchmark --hpo
  python -m tabulaforge.cli --dataset data.csv --target label --task classification
"""

from __future__ import annotations

import argparse
import json
import sys

from .core.config import Config
from .data.loaders import load_csv
from .models.factory import available_models
from .pipeline.tabula_pipeline import TabulaPipeline, benchmark


def _print_classification(metrics_list: list) -> None:
    header = ["Model", "ROC-AUC", "Acc", "F1", "P", "R", "HPO"]
    print("\n" + " | ".join(f"{h:<9}" for h in header))
    print("-" * (len(header) * 11))
    for m in metrics_list:
        mt = m.metrics
        print(
            " | ".join(
                [
                    f"{m.name:<9}",
                    f"{mt.get('roc_auc', float('nan')):<9.4f}",
                    f"{mt.get('accuracy', float('nan')):<9.4f}",
                    f"{mt.get('f1_macro', float('nan')):<9.4f}",
                    f"{mt.get('precision_macro', float('nan')):<9.4f}",
                    f"{mt.get('recall_macro', float('nan')):<9.4f}",
                    f"{'Y' if m.hpo else '-':<9}",
                ]
            )
        )


def _print_regression(metrics_list: list) -> None:
    header = ["Model", "R2", "RMSE", "MAE", "HPO"]
    print("\n" + " | ".join(f"{h:<9}" for h in header))
    print("-" * (len(header) * 11))
    for m in metrics_list:
        mt = m.metrics
        print(
            " | ".join(
                [
                    f"{m.name:<9}",
                    f"{mt.get('r2', float('nan')):<9.4f}",
                    f"{mt.get('rmse', float('nan')):<9.4f}",
                    f"{mt.get('mae', float('nan')):<9.4f}",
                    f"{'Y' if m.hpo else '-':<9}",
                ]
            )
        )


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tabulaforge", description="TabulaForge AutoML")
    parser.add_argument("--task", default="classification",
                        choices=["classification", "regression"])
    parser.add_argument("--model", default="xgboost")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--hpo", action="store_true", help="启用超参搜索")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--n-classes", type=int, default=2)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    config = Config(
        task=args.task,
        model=args.model,
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_classes=args.n_classes,
        noise=args.noise,
        random_state=args.random_state,
        n_trials=args.n_trials,
        use_hpo=args.hpo,
    )

    if args.dataset:
        if not args.target:
            print("ERROR: --dataset 需配合 --target", file=sys.stderr)
            return 2
        ds = load_csv(args.dataset, target_column=args.target, task=args.task)
    else:
        ds = None

    if args.benchmark:
        names = args.models or available_models()
        metrics = benchmark(names, config=config, dataset=ds, use_hpo=args.hpo)
        if args.json:
            print(json.dumps([m.as_dict() for m in metrics], ensure_ascii=False))
        else:
            if args.task == "classification":
                _print_classification(metrics)
            else:
                _print_regression(metrics)
        return 0

    pipe = TabulaPipeline(config)
    metrics = pipe.run(dataset=ds)
    if args.json:
        print(json.dumps(metrics.as_dict(), ensure_ascii=False))
    else:
        if args.task == "classification":
            _print_classification([metrics])
        else:
            _print_regression([metrics])
    return 0


if __name__ == "__main__":
    sys.exit(main())
