"""端到端 AutoML 流水线 + 横向 benchmark。作者：晨星"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..core.config import Config
from ..core.types import DatasetFrame, EvalMetrics, TaskType
from ..data.loaders import (
    load_csv,
    make_classification,
    make_regression,
    split_train_test,
)
from ..eval.metrics import Evaluator
from ..hpo.searcher import HpoSearcher
from ..models.factory import available_models, build_default_ensemble, build_model
from ..training.fit import Trainer


class TabulaPipeline:
    """单一模型端到端流水线。"""

    def __init__(
        self,
        config: Config,
        model: Optional[object] = None,
        hpo: Optional[bool] = None,
    ) -> None:
        self.config = config
        self.hpo = config.use_hpo if hpo is None else hpo
        if model is None:
            model = build_model(
                config.model,
                task=config.task,
                n_classes=config.n_classes,
            )
        self.model = model
        self.evaluator = Evaluator()

    def _ensure_dataset(self, dataset: Optional[DatasetFrame]) -> DatasetFrame:
        if dataset is not None:
            return dataset
        if self.config.task == TaskType.CLASSIFICATION.value:
            return make_classification(
                n_samples=self.config.n_samples,
                n_features=self.config.n_features,
                n_classes=self.config.n_classes,
                noise=self.config.noise,
                random_state=self.config.random_state,
            )
        return make_regression(
            n_samples=self.config.n_samples,
            n_features=self.config.n_features,
            noise=self.config.noise,
            random_state=self.config.random_state,
        )

    def run(self, dataset: Optional[DatasetFrame] = None) -> EvalMetrics:
        ds = self._ensure_dataset(dataset)
        train, test = split_train_test(
            ds, test_size=0.3, random_state=self.config.random_state
        )
        model = self.model
        if self.hpo:
            searcher = HpoSearcher(
                model.name,
                task=self.config.task,
                n_classes=train.n_classes,
                n_trials=self.config.n_trials,
                seed=self.config.random_state,
            )
            hpo_res = searcher.search(train)
            model = build_model(
                model.name,
                task=self.config.task,
                params=hpo_res.best_params,
                n_classes=train.n_classes,
            )
        Trainer().fit(model, train)
        res = model.predict(test.X)
        return self.evaluator.evaluate(res, test, hpo=self.hpo)

    def run_on_csv(self, path: str, target_column: str) -> EvalMetrics:
        ds = load_csv(path, target_column=target_column, task=self.config.task)
        return self.run(dataset=ds)


def benchmark(
    model_names: Optional[List[str]] = None,
    config: Optional[Config] = None,
    dataset: Optional[DatasetFrame] = None,
    use_hpo: bool = False,
) -> List[EvalMetrics]:
    """跨多个模型统一评测，按主指标降序返回。"""
    config = config or Config()
    if model_names is None:
        model_names = available_models()
    if dataset is None:
        if config.task == TaskType.CLASSIFICATION.value:
            dataset = make_classification(
                n_samples=config.n_samples,
                n_features=config.n_features,
                n_classes=config.n_classes,
                noise=config.noise,
                random_state=config.random_state,
            )
        else:
            dataset = make_regression(
                n_samples=config.n_samples,
                n_features=config.n_features,
                noise=config.noise,
                random_state=config.random_state,
            )

    train, test = split_train_test(
        dataset, test_size=0.3, random_state=config.random_state
    )
    evaluator = Evaluator()
    results: List[EvalMetrics] = []

    for name in model_names:
        try:
            n_cls = train.n_classes
            base = build_model(name, task=config.task, n_classes=n_cls)
        except Exception:  # noqa: BLE001
            continue

        if use_hpo:
            try:
                hpo = HpoSearcher(
                    name, task=config.task, n_classes=n_cls,
                    n_trials=config.n_trials, seed=config.random_state,
                ).search(train)
                model = build_model(
                    name, task=config.task, params=hpo.best_params, n_classes=n_cls
                )
            except Exception:  # noqa: BLE001
                model = base
        else:
            model = base

        try:
            Trainer().fit(model, train)
            res = model.predict(test.X)
            metrics = evaluator.evaluate(res, test, hpo=use_hpo)
        except Exception:  # noqa: BLE001
            continue
        results.append(metrics)

    results.sort(key=lambda m: m.primary(), reverse=True)
    return results
