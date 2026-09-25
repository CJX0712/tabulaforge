"""错误码体系（E100~E500）。作者：晨星"""

from __future__ import annotations


class TabulaForgeError(Exception):
    """所有 TabulaForge 错误的基类。"""

    code: str = "E000"

    def __init__(self, message: str, *, code: str | None = None, cause: Exception | None = None):
        self.code = code or self.__class__.code
        self.message = message
        self.cause = cause
        super().__init__(f"[{self.code}] {message}")
        if cause is not None:
            self.__cause__ = cause


class DataError(TabulaForgeError):
    code = "E100"


class ModelError(TabulaForgeError):
    code = "E200"


class TrainingError(TabulaForgeError):
    code = "E300"


class EvalError(TabulaForgeError):
    code = "E400"


class ConfigError(TabulaForgeError):
    code = "E500"


__all__ = [
    "TabulaForgeError",
    "DataError",
    "ModelError",
    "TrainingError",
    "EvalError",
    "ConfigError",
]
