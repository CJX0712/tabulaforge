"""pipeline 层单测。作者：晨星"""

from tabulaforge.core.config import Config
from tabulaforge.core.types import TaskType
from tabulaforge.data.loaders import make_classification, make_regression
from tabulaforge.models.factory import build_model
from tabulaforge.pipeline.tabula_pipeline import TabulaPipeline, benchmark


def test_pipeline_run_classification():
    cfg = Config(task="classification", model="xgboost", random_state=11)
    pipe = TabulaPipeline(cfg)
    m = pipe.run()
    assert m.task == "classification"
    assert "accuracy" in m.metrics
    assert m.metrics["accuracy"] > 0.7


def test_pipeline_run_regression():
    cfg = Config(task="regression", model="lightgbm", random_state=12)
    pipe = TabulaPipeline(cfg)
    m = pipe.run()
    assert m.task == "regression"
    assert "r2" in m.metrics


def test_benchmark_sorted_descending():
    cfg = Config(task="classification", random_state=13, n_samples=800, n_features=10)
    metrics = benchmark(config=cfg)
    assert len(metrics) >= 2
    prim = [m.primary() for m in metrics]
    assert prim == sorted(prim, reverse=True)


def test_benchmark_hpo_small():
    cfg = Config(task="classification", random_state=14, n_trials=5)
    metrics = benchmark(["xgboost", "sklearn"], config=cfg, use_hpo=True)
    assert len(metrics) == 2
    assert all(m.hpo for m in metrics)


def test_pipeline_with_external_dataset():
    ds = make_classification(random_state=15)
    cfg = Config(task="classification", model="sklearn", random_state=15)
    pipe = TabulaPipeline(cfg)
    m = pipe.run(dataset=ds)
    assert m.metrics["accuracy"] > 0.7
