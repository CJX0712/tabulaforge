"""模型基类：统一 fit / predict / predict_proba 接口。作者：晨星"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..core.errors import ModelError
from ..core.types import ModelResult, TaskType


class BaseModel:
    """所有模型包装器的基类。

    子类实现 `_build()` 返回已实例化的 sklearn/ xgboost/ lightgbm estimator。
    基类负责统一的 fit / predict（含分类概率）逻辑。
    """

    name: str = "base"

    def __init__(
        self, task: str, params: Optional[Dict[str, Any]] = None, n_classes: int = 2
    ) -> None:
        self.task = TaskType(task)
        self.params: Dict[str, Any] = dict(params or {})
        self.n_classes = n_classes
        self._model = None
        self._fitted = False

    def _build(self):
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        try:
            self._model = self._build()
            self._model.fit(X, y)
        except Exception as e:  # noqa: BLE001
            raise ModelError(f"{self.name} 拟合失败", cause=e) from e
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> ModelResult:
        if not self._fitted or self._model is None:
            raise ModelError(f"{self.name} 未拟合即 predict")
        X = np.asarray(X, dtype=np.float64)
        try:
            pred = np.asarray(self._model.predict(X))
        except Exception as e:  # noqa: BLE001
            raise ModelError(f"{self.name} 推理失败", cause=e) from e
        proba = None
        if self.task == TaskType.CLASSIFICATION and hasattr(self._model, "predict_proba"):
            try:
                proba = np.asarray(self._model.predict_proba(X), dtype=np.float64)
            except Exception:  # noqa: BLE001
                proba = None
        return ModelResult(name=self.name, pred=pred, proba=proba)

    def get_default_param_space(self) -> Dict[str, Any]:
        """返回 HPO 默认参数空间（Optuna/RandSearch 通用 schema）。"""
        return {}
