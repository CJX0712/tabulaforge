"""core 层单测。作者：晨星"""

import numpy as np
import pytest

from tabulaforge.core.config import Config
from tabulaforge.core.errors import ConfigError, DataError, ModelError
from tabulaforge.core.types import DatasetFrame, TaskType


def test_datasetframe_validation():
    X = np.random.default_rng(0).normal(size=(100, 5))
    y = np.zeros(100)
    ds = DatasetFrame(X=X, y=y, task="classification")
    assert ds.n_samples == 100
    assert ds.n_features == 5
    assert ds.task == TaskType.CLASSIFICATION


def test_datasetframe_mismatch_raises():
    with pytest.raises(ValueError):
        DatasetFrame(X=np.zeros((10, 3)), y=np.zeros(9))


def test_datasetframe_2d_required():
    with pytest.raises(ValueError):
        DatasetFrame(X=np.zeros(10), y=np.zeros(10))


def test_datasetframe_task_invalid():
    with pytest.raises(ValueError):
        DatasetFrame(X=np.zeros((10, 3)), y=np.zeros(10), task="unknown_task")


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("TABULAFORGE_TASK", "regression")
    monkeypatch.setenv("TABULAFORGE_MODEL", "lightgbm")
    monkeypatch.setenv("TABULAFORGE_N_TRIALS", "50")
    c = Config(env_override=True)
    assert c.task == "regression"
    assert c.model == "lightgbm"
    assert c.n_trials == 50


def test_config_illegal_task():
    with pytest.raises(ConfigError):
        Config(task="not_a_task")


def test_errors_have_codes():
    assert DataError("x").code == "E100"
    assert ModelError("x").code == "E200"
    assert ConfigError("x").code == "E500"
