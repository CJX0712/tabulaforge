"""训练编排：将模型拟合到数据集并计时。作者：晨星"""

from __future__ import annotations

import time

from ..core.errors import TrainingError
from ..core.types import DatasetFrame, FitResult


class Trainer:
    """模型训练编排器。"""

    def fit(self, model: object, dataset: DatasetFrame) -> FitResult:
        try:
            t0 = time.perf_counter()
            model.fit(dataset.X, dataset.y)
            elapsed = time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001
            raise TrainingError(
                f"训练 {getattr(model, 'name', '?')} 失败", cause=e
            ) from e
        return FitResult(
            name=getattr(model, "name", "model"),
            fitted=True,
            n_train=dataset.n_samples,
            elapsed_sec=elapsed,
        )
