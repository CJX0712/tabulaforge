"""models 层单测。作者：晨星"""

import numpy as np
import pytest

from tabulaforge.core.errors import ModelError
from tabulaforge.core.types import TaskType
from tabulaforge.data.loaders import make_classification, make_regression
from tabulaforge.models.factory import (
    available_models,
    build_default_ensemble,
    build_model,
)
from tabulaforge.training.fit import Trainer

CLS = make_classification(n_samples=600, n_features=10, n_classes=2, random_state=0)
REG = make_regression(n_samples=600, n_features=10, random_state=0)


def test_available_models_nonempty():
    assert len(available_models()) >= 1


def test_build_unknown_raises():
    with pytest.raises(ModelError):
        build_model("does_not_exist", task="classification")


def test_build_default_ensemble_nonempty():
    ens = build_default_ensemble(task="classification", n_classes=2)
    assert len(ens) >= 1


@pytest.mark.parametrize("name", available_models())
def test_model_classification_fit_predict(name):
    ds = CLS
    n_cls = ds.n_classes
    model = build_model(name, task="classification", n_classes=n_cls)
    Trainer().fit(model, ds)
    res = model.predict(ds.X)
    assert res.pred.shape[0] == ds.n_samples
    if ds.task == TaskType.CLASSIFICATION and name in ("xgboost", "lightgbm", "sklearn"):
        # 这些模型支持概率
        assert res.proba is not None and res.proba.shape[0] == ds.n_samples


@pytest.mark.parametrize("name", available_models())
def test_model_regression_fit_predict(name):
    ds = REG
    model = build_model(name, task="regression", n_classes=1)
    Trainer().fit(model, ds)
    res = model.predict(ds.X)
    assert res.pred.shape[0] == ds.n_samples
    assert res.proba is None
