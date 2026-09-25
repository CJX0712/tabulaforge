"""data 层单测。作者：晨星"""

import numpy as np
import pytest  # noqa: E402

from tabulaforge.core.types import TaskType
from tabulaforge.data.loaders import (
    make_classification,
    make_regression,
    split_train_test,
)


def test_make_classification_shape():
    ds = make_classification(n_samples=1000, n_features=12, n_classes=2, random_state=1)
    assert ds.X.shape == (1000, 12)
    assert ds.task == TaskType.CLASSIFICATION
    assert ds.n_classes == 2


def test_make_regression_shape():
    ds = make_regression(n_samples=800, n_features=8, random_state=2)
    assert ds.X.shape == (800, 8)
    assert ds.task == TaskType.REGRESSION


def test_make_reproducible():
    a = make_classification(random_state=7)
    b = make_classification(random_state=7)
    assert np.allclose(a.X, b.X)
    assert np.array_equal(a.y, b.y)


def test_split_train_test():
    ds = make_classification(n_samples=500, random_state=3)
    train, test = split_train_test(ds, test_size=0.3, random_state=3)
    assert train.n_samples + test.n_samples == 500
    assert test.n_samples == 150
