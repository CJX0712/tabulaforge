# TabulaForge Makefile — 作者：晨星
PYTHON ?= python3

.PHONY: help venv install test demo benchmark cli clean

help:
	@echo "Targets:"
	@echo "  make venv      创建虚拟环境并安装依赖"
	@echo "  make install   安装依赖"
	@echo "  make test      运行 pytest"
	@echo "  make demo      运行端到端演示"
	@echo "  make benchmark 运行横向评测 (CLI)"
	@echo "  make cli       运行命令行入口"
	@echo "  make clean     清理缓存"

venv:
	$(PYTHON) -m venv .venv
	.venv/Scripts/activate || . .venv/bin/activate
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -r requirements.txt

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

demo:
	$(PYTHON) -m tabulaforge.examples.run_demo

benchmark:
	$(PYTHON) -m tabulaforge.cli --benchmark

cli:
	$(PYTHON) -m tabulaforge.cli --task classification --model xgboost

clean:
	rm -rf .pytest_cache __pycache__ .venv
	find . -name '__pycache__' -type d -exec rm -rf {} +
