# TabulaForge 容器镜像 — 作者：晨星
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 默认运行 demo
CMD ["python", "-m", "tabulaforge.examples.run_demo"]
