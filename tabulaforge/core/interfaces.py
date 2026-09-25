"""模块接口契约（Protocol）。作者：晨星"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np

from .types import DatasetFrame, EvalMetrics, FitResult, HpoResult, ModelResult


@runtime_checkable
class IDataLoader(Protocol):
    def load(self, **kwargs: Any) -> DatasetFrame: ...


@runtime_checkable
class IModel(Protocol):
    name: str

    def fit(self, X: np.ndarray, y: np.ndarray) -> "IModel":
        ...

    def predict(self, X: np.ndarray) -> ModelResult:
        ...


@runtime_checkable
class IHPOSearcher(Protocol):
    def search(self, dataset: DatasetFrame) -> HpoResult:
        ...


@runtime_checkable
class ITrainer(Protocol):
    def fit(self, model: Any, dataset: DatasetFrame) -> FitResult:
        ...


@runtime_checkable
class IEvaluator(Protocol):
    def evaluate(self, result: ModelResult, dataset: DatasetFrame) -> EvalMetrics:
        ...


@runtime_checkable
class IPipeline(Protocol):
    def run(self, dataset: Optional[DatasetFrame] = None) -> EvalMetrics:
        ...
