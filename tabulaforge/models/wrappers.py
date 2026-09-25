"""顶级开源模型封装：XGBoost / LightGBM / scikit-learn 兜底。作者：晨星

- XGBoostModel: 封装 xgboost.XGBClassifier / XGBRegressor
- LightGBMModel: 封装 lightgbm.LGBMClassifier / LGBMRegressor
- SklearnModel: 纯 scikit-learn HistGradientBoosting（零额外依赖兜底）
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..core.types import TaskType
from .base import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost 封装（顶级梯度提升库）。"""

    name = "xgboost"

    def _build(self):
        try:
            import xgboost as xgb
        except ImportError as e:
            raise RuntimeError(
                "xgboost 未安装；可改用 sklearn 兜底模型（如 'sklearn'）"
            ) from e
        if self.task == TaskType.CLASSIFICATION:
            return xgb.XGBClassifier(
                objective="binary:logistic" if self.n_classes == 2 else "multi:softprob",
                eval_metric="logloss",
                random_state=42,
                n_jobs=1,
                **self.params,
            )
        return xgb.XGBRegressor(random_state=42, n_jobs=1, **self.params)

    def get_default_param_space(self) -> Dict[str, Any]:
        return {
            "n_estimators": ("int", 50, 400),
            "max_depth": ("int", 3, 10),
            "learning_rate": ("log", 0.01, 0.3),
            "subsample": ("float", 0.6, 1.0),
            "colsample_bytree": ("float", 0.6, 1.0),
            "reg_lambda": ("float", 0.0, 10.0),
        }


class LightGBMModel(BaseModel):
    """LightGBM 封装（微软顶级梯度提升库）。"""

    name = "lightgbm"

    def _build(self):
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise RuntimeError(
                "lightgbm 未安装；可改用 sklearn 兜底模型（如 'sklearn'）"
            ) from e
        if self.task == TaskType.CLASSIFICATION:
            n_cls = self.n_classes
            return lgb.LGBMClassifier(
                objective="binary" if n_cls == 2 else "multiclass",
                random_state=42,
                n_jobs=1,
                verbose=-1,
                **self.params,
            )
        return lgb.LGBMRegressor(random_state=42, n_jobs=1, verbose=-1, **self.params)

    def get_default_param_space(self) -> Dict[str, Any]:
        return {
            "n_estimators": ("int", 50, 400),
            "max_depth": ("int", 3, 10),
            "learning_rate": ("log", 0.01, 0.3),
            "num_leaves": ("int", 15, 63),
            "subsample": ("float", 0.6, 1.0),
            "colsample_bytree": ("float", 0.6, 1.0),
            "reg_lambda": ("float", 0.0, 10.0),
        }


class SklearnModel(BaseModel):
    """纯 scikit-learn 兜底（HistGradientBoosting，无需 XGBoost/LightGBM）。"""

    name = "sklearn"

    def _build(self):
        try:
            from sklearn.ensemble import (
                HistGradientBoostingClassifier,
                HistGradientBoostingRegressor,
            )
        except ImportError as e:
            raise RuntimeError("scikit-learn 未安装") from e
        if self.task == TaskType.CLASSIFICATION:
            return HistGradientBoostingClassifier(random_state=42, **self.params)
        return HistGradientBoostingRegressor(random_state=42, **self.params)

    def get_default_param_space(self) -> Dict[str, Any]:
        return {
            "max_iter": ("int", 50, 400),
            "learning_rate": ("log", 0.01, 0.3),
            "max_depth": ("int", 3, 10),
            "l2_regularization": ("float", 0.0, 10.0),
        }
