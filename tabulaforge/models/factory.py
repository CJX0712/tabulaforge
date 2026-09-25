"""模型工厂：按名称构造模型，含可用性探测与兜底。作者：晨星"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core.errors import ModelError
from ..core.types import TaskType
from .wrappers import LightGBMModel, SklearnModel, XGBoostModel

_MODELS = {
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
    "sklearn": SklearnModel,
}

# 默认集成顺序：顶级 SOTA 在前，sklearn 兜底在后
DEFAULT_ENSEMBLE = ["xgboost", "lightgbm", "sklearn"]


def _available(name: str) -> bool:
    if name == "xgboost":
        try:
            import xgboost  # noqa: F401

            return True
        except ImportError:
            return False
    if name == "lightgbm":
        try:
            import lightgbm  # noqa: F401

            return True
        except ImportError:
            return False
    if name == "sklearn":
        try:
            import sklearn  # noqa: F401

            return True
        except ImportError:
            return False
    return False


def available_models() -> List[str]:
    """返回当前可用模型名。"""
    return [n for n in _MODELS if _available(n)]


def build_model(
    name: str,
    task: str = "classification",
    params: Optional[Dict[str, Any]] = None,
    n_classes: int = 2,
) -> Any:
    """按名称构造模型。

    - 指定模型可用则构造；
    - 指定模型不可用抛 ModelError（由调用方决定跳过或兜底）。
    """
    name = name.lower()
    if name not in _MODELS:
        raise ModelError(f"未知模型: {name}；可选: {available_models()}")
    if not _available(name):
        raise ModelError(f"模型 {name} 后端不可用（未安装对应库）")
    p = dict(params or {})
    return _MODELS[name](task=task, params=p, n_classes=n_classes)


def build_default_ensemble(
    task: str = "classification", n_classes: int = 2
) -> List[Any]:
    """构造默认模型集成（跳过不可用后端，至少保留 sklearn 兜底）。"""
    models: List[Any] = []
    for n in DEFAULT_ENSEMBLE:
        if not _available(n):
            continue
        try:
            models.append(build_model(n, task=task, n_classes=n_classes))
        except ModelError:
            continue
    if not models:
        # 极端兜底：直接尝试 sklearn
        models = [build_model("sklearn", task=task, n_classes=n_classes)]
    return models
