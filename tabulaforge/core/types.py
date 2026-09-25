"""核心数据类型定义。作者：晨星

所有跨模块数据载体用 dataclass 描述。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class TaskType(str, Enum):
    """学习任务类型。"""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


@dataclass
class DatasetFrame:
    """一份表格数据集。

    X: 特征矩阵 (n_samples, n_features)
    y: 标签/目标 (n_samples,)
    task: 任务类型
    feature_names / target_name: 可选列名
    """

    X: np.ndarray
    y: np.ndarray
    task: TaskType = TaskType.CLASSIFICATION
    feature_names: Optional[List[str]] = None
    target_name: Optional[str] = None

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=np.float64).ravel()
        if self.X.ndim != 2:
            raise ValueError("X 必须是 2D 矩阵 (n_samples, n_features)")
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X 与 y 样本数不一致")
        if isinstance(self.task, str):
            self.task = TaskType(self.task)

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def n_classes(self) -> int:
        return int(np.unique(self.y).shape[0])


@dataclass
class ModelResult:
    """单个模型在数据集上的输出。

    pred: 预测值
    proba: 分类概率 (n_samples, n_classes)，回归为 None
    meta: 附加信息
    """

    name: str
    pred: np.ndarray
    proba: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.pred = np.asarray(self.pred).ravel()
        if self.proba is not None:
            self.proba = np.asarray(self.proba)


@dataclass
class FitResult:
    """训练结果。"""

    name: str
    fitted: bool
    n_train: int
    elapsed_sec: float
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HpoResult:
    """超参搜索结果。"""

    name: str
    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    direction: str = "maximize"
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalMetrics:
    """评测指标集合（分类与回归共用，按需填充）。"""

    name: str
    task: str
    metrics: Dict[str, float]
    hpo: bool = False

    def primary(self) -> float:
        """用于排序的主指标：分类取 roc_auc/accuracy，回归取 r2。"""
        for k in ("roc_auc", "accuracy", "r2"):
            if k in self.metrics:
                return self.metrics[k]
        # 回归时 r2 可能为负，回退到 -rmse
        if "rmse" in self.metrics:
            return -self.metrics["rmse"]
        return 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "task": self.task,
            "hpo": self.hpo,
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
        }
