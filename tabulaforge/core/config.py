"""配置与运行时参数（支持 ANOMALYFORGE 风格环境变量覆盖）。作者：晨星"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict

from .errors import ConfigError
from .types import TaskType


@dataclass
class Config:
    """TabulaForge 全局配置。"""

    task: str = "classification"
    model: str = "xgboost"
    contamination: float = 0.1  # 仅占位兼容
    random_state: int = 42
    n_samples: int = 2000
    n_features: int = 20
    n_classes: int = 2
    noise: float = 0.1
    n_trials: int = 30
    use_hpo: bool = False
    env_override: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.env_override:
            self._apply_env()
        # 校验 task
        try:
            self.task = TaskType(self.task).value
        except ValueError as e:
            raise ConfigError(f"非法 task: {self.task}", cause=e) from e

    def _apply_env(self) -> None:
        if v := os.getenv("TABULAFORGE_TASK"):
            self.task = v
        if v := os.getenv("TABULAFORGE_MODEL"):
            self.model = v
        if v := os.getenv("TABULAFORGE_RANDOM_STATE"):
            self.random_state = int(v)
        if v := os.getenv("TABULAFORGE_N_SAMPLES"):
            self.n_samples = int(v)
        if v := os.getenv("TABULAFORGE_N_FEATURES"):
            self.n_features = int(v)
        if v := os.getenv("TABULAFORGE_N_TRIALS"):
            self.n_trials = int(v)
        if v := os.getenv("TABULAFORGE_USE_HPO"):
            self.use_hpo = str(v).lower() in ("1", "true", "yes")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "model": self.model,
            "random_state": self.random_state,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "noise": self.noise,
            "n_trials": self.n_trials,
            "use_hpo": self.use_hpo,
            "extra": self.extra,
        }
