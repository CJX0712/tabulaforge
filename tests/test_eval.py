"""eval 层单测。作者：晨星"""

import numpy as np

from tabulaforge.core.types import ModelResult, TaskType
from tabulaforge.data.loaders import make_classification, make_regression
from tabulaforge.eval.metrics import Evaluator
from tabulaforge.models.factory import build_model
from tabulaforge.training.fit import Trainer

CLS = make_classification(n_samples=600, n_features=10, n_classes=2, random_state=1)
REG = make_regression(n_samples=600, n_features=10, random_state=1)


def test_eval_classification_keys():
    model = build_model("xgboost", task="classification", n_classes=2)
    Trainer().fit(model, CLS)
    res = model.predict(CLS.X)
    m = Evaluator().evaluate(res, CLS)
    assert m.task == "classification"
    for k in ("accuracy", "f1_macro", "roc_auc"):
        assert k in m.metrics
    assert 0.0 <= m.metrics["accuracy"] <= 1.0


def test_eval_regression_keys():
    model = build_model("xgboost", task="regression", n_classes=1)
    Trainer().fit(model, REG)
    res = model.predict(REG.X)
    m = Evaluator().evaluate(res, REG)
    assert m.task == "regression"
    for k in ("rmse", "mae", "r2"):
        assert k in m.metrics


def test_eval_perfect_classification():
    y = CLS.y
    # 构造完美预测（直接返回真实标签）与概率
    proba = np.zeros((len(y), 2))
    proba[:, 1] = y
    proba[:, 0] = 1 - y
    res = ModelResult(name="perfect", pred=y.copy(), proba=proba)
    m = Evaluator().evaluate(res, CLS)
    assert abs(m.metrics["accuracy"] - 1.0) < 1e-6
    assert abs(m.metrics["roc_auc"] - 1.0) < 1e-6
