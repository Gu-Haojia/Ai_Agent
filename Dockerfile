FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=UTC

# 安装系统依赖与 psql 工具，便于健康检查；tzdata 用于时区同步
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    build-essential \
    libpq-dev \
    postgresql-client \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 仅拷贝依赖清单，保持镜像精简
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 提前放入入口脚本，运行时仍会被挂载的项目代码覆盖
COPY docker/entrypoint.sh /app/docker/entrypoint.sh
RUN chmod +x /app/docker/entrypoint.sh

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["python", "qq_group_bot.py"]
