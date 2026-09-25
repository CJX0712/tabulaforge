"""超参搜索：Optuna 优先，缺失时 numpy 随机搜索兜底。作者：晨星

对所有模型统一：从 `model.get_default_param_space()` 读取参数空间，
在训练集内部 80/20 划分上评估主指标，最大化该指标。
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

import numpy as np

from ..core.errors import ModelError
from ..core.types import DatasetFrame, HpoResult, TaskType
from ..models.factory import build_model

try:
    import optuna

    _OPTUNA_OK = True
except ImportError:  # pragma: no cover
    _OPTUNA_OK = False


def _score_on_split(
    model_name: str,
    task: str,
    n_classes: int,
    params: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """在 (train, val) 上训练并返回用于最大化的主指标。"""
    try:
        model = build_model(model_name, task=task, params=params, n_classes=n_classes)
        model.fit(X_train, y_train)
        res = model.predict(X_val)
    except Exception:  # noqa: BLE001
        return -1e9  # 非法参数组合

    from sklearn.metrics import (
        accuracy_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )

    if task == TaskType.CLASSIFICATION.value:
        if res.proba is not None and res.proba.shape[1] == 2:
            try:
                return float(roc_auc_score(y_val, res.proba[:, 1]))
            except Exception:  # noqa: BLE001
                pass
        return float(accuracy_score(y_val, res.pred))
    # 回归：最大化 r2
    return float(r2_score(y_val, res.pred))


class HpoSearcher:
    """对指定模型做超参搜索。"""

    def __init__(
        self,
        model_name: str,
        task: str = "classification",
        n_classes: int = 2,
        n_trials: int = 30,
        timeout: Optional[float] = None,
        seed: int = 42,
    ) -> None:
        self.model_name = model_name
        self.task = TaskType(task).value
        self.n_classes = n_classes
        self.n_trials = n_trials
        self.timeout = timeout
        self.seed = seed
        self.backend = "optuna" if _OPTUNA_OK else "random"

    def search(self, dataset: DatasetFrame) -> HpoResult:
        # 训练集内部再划分 80/20 作为 HPO 验证
        rng = np.random.default_rng(self.seed)
        n = dataset.n_samples
        idx = rng.permutation(n)
        k = max(2, int(round(n * 0.2)))
        val_idx, tr_idx = idx[:k], idx[k:]
        X_tr, y_tr = dataset.X[tr_idx], dataset.y[tr_idx]
        X_val, y_val = dataset.X[val_idx], dataset.y[val_idx]

        # 取一个示例参数空间
        probe = build_model(
            self.model_name, task=self.task, n_classes=self.n_classes
        )
        space = probe.get_default_param_space()

        if self.backend == "optuna":
            return self._search_optuna(space, X_tr, y_tr, X_val, y_val)
        return self._search_random(space, X_tr, y_tr, X_val, y_val)

    def _sample(self, space: Dict[str, Any], rng: np.random.Generator, trial=None):
        params: Dict[str, Any] = {}
        for name, spec in space.items():
            kind, low, high = spec
            if trial is not None:
                if kind == "int":
                    params[name] = trial.suggest_int(name, low, high)
                elif kind == "float":
                    params[name] = trial.suggest_float(name, low, high)
                else:  # log
                    params[name] = trial.suggest_float(name, low, high, log=True)
            else:
                if kind == "int":
                    params[name] = int(rng.integers(low, high + 1))
                elif kind == "float":
                    params[name] = float(rng.uniform(low, high))
                else:  # log-uniform
                    params[name] = float(
                        np.exp(rng.uniform(np.log(low), np.log(high)))
                    )
        return params

    def _search_optuna(self, space, X_tr, y_tr, X_val, y_val) -> HpoResult:
        import optuna

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )

        def objective(trial):
            params = self._sample(space, None, trial)
            return _score_on_split(
                self.model_name, self.task, self.n_classes,
                params, X_tr, y_tr, X_val, y_val,
            )

        study.optimize(self._objective_safe(objective), n_trials=self.n_trials, timeout=self.timeout)
        return HpoResult(
            name=self.model_name,
            best_params=dict(study.best_params),
            best_score=float(study.best_value),
            n_trials=len(study.trials),
            direction="maximize",
            meta={"backend": "optuna"},
        )

    def _objective_safe(self, objective):
        def _wrap(trial):
            try:
                return objective(trial)
            except Exception:  # noqa: BLE001
                return -1e9

        return _wrap

    def _search_random(self, space, X_tr, y_tr, X_val, y_val) -> HpoResult:
        rng = np.random.default_rng(self.seed)
        best_score = -1e9
        best_params: Dict[str, Any] = {}
        for _ in range(self.n_trials):
            params = self._sample(space, rng, None)
            s = _score_on_split(
                self.model_name, self.task, self.n_classes,
                params, X_tr, y_tr, X_val, y_val,
            )
            if s > best_score:
                best_score, best_params = s, params
        return HpoResult(
            name=self.model_name,
            best_params=best_params,
            best_score=float(best_score),
            n_trials=self.n_trials,
            direction="maximize",
            meta={"backend": "random"},
        )
