"""TabulaForge — 模块化自动化表格机器学习（AutoML Tabular）系统。

顶层导出核心公共 API。作者：晨星
"""

from .core.types import (
    DatasetFrame,
    ModelResult,
    EvalMetrics,
    HpoResult,
    TaskType,
)
from .core.errors import (
    TabulaForgeError,
    DataError,
    ModelError,
    TrainingError,
    EvalError,
    ConfigError,
)
from .core.config import Config
from .pipeline.tabula_pipeline import TabulaPipeline

__version__ = "0.1.0"
__author__ = "晨星"

__all__ = [
    "DatasetFrame",
    "ModelResult",
    "EvalMetrics",
    "HpoResult",
    "TaskType",
    "TabulaForgeError",
    "DataError",
    "ModelError",
    "TrainingError",
    "EvalError",
    "ConfigError",
    "Config",
    "TabulaPipeline",
    "__version__",
    "__author__",
]
