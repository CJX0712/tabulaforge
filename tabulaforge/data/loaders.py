"""数据载入与合成数据集生成。

- load_csv / load_parquet：从文件载入（pandas，缺失时回退 numpy）
- make_classification / make_regression：基于 scikit-learn 的合成数据生成器
- split_train_test：随机切分
作者：晨星
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from ..core.errors import DataError
from ..core.types import DatasetFrame, TaskType


def make_classification(
    n_samples: int = 2000,
    n_features: int = 20,
    n_classes: int = 2,
    noise: float = 0.1,
    random_state: int = 42,
) -> DatasetFrame:
    """生成合成分类数据集（scikit-learn）。"""
    try:
        from sklearn.datasets import make_classification as _mc
    except ImportError as e:
        raise DataError("需要 scikit-learn 生成合成数据", cause=e) from e

    n_informative = max(2, min(n_features // 2, n_features))
    X, y = _mc(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=max(0, n_features - n_informative - 2),
        n_classes=n_classes,
        flip_y=noise,
        random_state=random_state,
    )
    feature_names = [f"f{i}" for i in range(n_features)]
    return DatasetFrame(
        X=X,
        y=y.astype(np.float64),
        task=TaskType.CLASSIFICATION,
        feature_names=feature_names,
        target_name="label",
    )


def make_regression(
    n_samples: int = 2000,
    n_features: int = 20,
    noise: float = 0.1,
    random_state: int = 42,
) -> DatasetFrame:
    """生成合成回归数据集（scikit-learn）。"""
    try:
        from sklearn.datasets import make_regression as _mr
    except ImportError as e:
        raise DataError("需要 scikit-learn 生成合成数据", cause=e) from e

    n_informative = max(2, min(n_features // 2, n_features))
    X, y = _mr(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=random_state,
    )
    feature_names = [f"f{i}" for i in range(n_features)]
    return DatasetFrame(
        X=X,
        y=y.astype(np.float64),
        task=TaskType.REGRESSION,
        feature_names=feature_names,
        target_name="target",
    )


def load_csv(path: str, target_column: str, task: str = "classification") -> DatasetFrame:
    """从 CSV 载入（pandas 优先，缺失回退 numpy）。"""
    if not os.path.exists(path):
        raise DataError(f"文件不存在: {path}")
    try:
        import pandas as pd

        df = pd.read_csv(path)
        if target_column not in df.columns:
            raise DataError(f"target_column 不存在: {target_column}")
        y = df[target_column].to_numpy(dtype=np.float64)
        X = df.drop(columns=[target_column]).to_numpy(dtype=np.float64)
        feature_names = [c for c in df.columns if c != target_column]
    except ImportError:
        import csv

        with open(path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = [r for r in reader if r]
        data = np.asarray(rows, dtype=np.float64)
        if target_column not in header:
            raise DataError(f"target_column 不存在: {target_column}")
        col = header.index(target_column)
        y = data[:, col]
        feat_cols = [i for i in range(len(header)) if i != col]
        X = data[:, feat_cols]
        feature_names = [header[i] for i in feat_cols]

    return DatasetFrame(
        X=X,
        y=y,
        task=TaskType(task),
        feature_names=feature_names,
        target_name=target_column,
    )


def split_train_test(
    dataset: DatasetFrame, test_size: float = 0.3, random_state: int = 42
) -> Tuple[DatasetFrame, DatasetFrame]:
    """按行随机切分。"""
    rng = np.random.default_rng(random_state)
    n = dataset.n_samples
    idx = rng.permutation(n)
    n_test = max(1, int(round(n * test_size)))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    train = DatasetFrame(
        X=dataset.X[train_idx],
        y=dataset.y[train_idx],
        task=dataset.task,
        feature_names=dataset.feature_names,
        target_name=dataset.target_name,
    )
    test = DatasetFrame(
        X=dataset.X[test_idx],
        y=dataset.y[test_idx],
        task=dataset.task,
        feature_names=dataset.feature_names,
        target_name=dataset.target_name,
    )
    return train, test
