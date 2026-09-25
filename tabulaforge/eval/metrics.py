"""评测指标：分类 + 回归。scikit-learn 优先，缺失时 numpy 兜底。作者：晨星"""

from __future__ import annotations

import numpy as np

from ..core.errors import EvalError
from ..core.types import DatasetFrame, EvalMetrics, ModelResult, TaskType

try:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
        roc_auc_score,
    )

    _SKLEARN_OK = True
except ImportError:  # pragma: no cover
    _SKLEARN_OK = False


def _roc_auc(y_true, proba_pos):
    """二分类 ROC-AUC（numpy 秩和兜底）。"""
    if _SKLEARN_OK:
        return float(roc_auc_score(y_true, proba_pos))
    y = np.asarray(y_true, int)
    s = np.asarray(proba_pos, float)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


class Evaluator:
    """对 ModelResult 进行端到端评测。"""

    def evaluate(
        self, result: ModelResult, dataset: DatasetFrame, hpo: bool = False
    ) -> EvalMetrics:
        y_true = dataset.y
        pred = result.pred
        task = dataset.task
        try:
            if task == TaskType.CLASSIFICATION:
                metrics = self._classification(result, y_true, pred)
            else:
                metrics = self._regression(y_true, pred)
        except Exception as e:  # noqa: BLE001
            raise EvalError(f"评测 {result.name} 失败", cause=e) from e
        return EvalMetrics(name=result.name, task=task.value, metrics=metrics, hpo=hpo)

    def _classification(self, result, y_true, pred) -> dict:
        if _SKLEARN_OK:
            m = {
                "accuracy": float(accuracy_score(y_true, pred)),
                "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
                "precision_macro": float(
                    precision_score(y_true, pred, average="macro", zero_division=0)
                ),
                "recall_macro": float(
                    recall_score(y_true, pred, average="macro", zero_division=0)
                ),
            }
            if result.proba is not None and result.proba.shape[1] == 2:
                m["roc_auc"] = _roc_auc(y_true, result.proba[:, 1])
            else:
                # 多分类用 macro OvR ROC-AUC
                try:
                    from sklearn.metrics import roc_auc_score as _ra

                    m["roc_auc"] = float(
                        _ra(y_true, result.proba, multi_class="ovr", average="macro")
                    ) if result.proba is not None else m["accuracy"]
                except Exception:  # noqa: BLE001
                    m["roc_auc"] = m["accuracy"]
            return m
        # numpy 兜底：仅 accuracy
        return {"accuracy": float(np.mean(np.asarray(y_true) == np.asarray(pred)))}

    def _regression(self, y_true, pred) -> dict:
        if _SKLEARN_OK:
            return {
                "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
                "mae": float(mean_absolute_error(y_true, pred)),
                "r2": float(r2_score(y_true, pred)),
            }
        # numpy 兜底
        err = np.asarray(y_true) - np.asarray(pred)
        return {
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
            "r2": (
                float(1 - np.sum(err ** 2) / (np.var(y_true) * len(y_true) + 1e-12))
            ),
        }
