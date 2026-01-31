#!/usr/bin/env bash
set -euo pipefail

# 统一切换到应用目录
APP_DIR="/app"
cd "${APP_DIR}"

# 每次启动自动加载最新 .env（若存在）
load_env_file() {
  if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    . ".env"
    set +a
  fi
}

# 需要存在的目录与文件，确保宿主机可读写
REQUIRED_DIRS=("logs" "prompts" "ticket_data" "images")
REQUIRED_FILES=(".qq_group_threads.json" ".qq_reminder_threads.json" ".qq_group_memnames.json" ".meru_watch.json")

create_dirs() {
  for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "${APP_DIR}/${dir}" ]; then
      mkdir -p "${APP_DIR}/${dir}"
    fi
    chmod 755 "${APP_DIR}/${dir}"
  done
}

create_files() {
  for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "${APP_DIR}/${file}" ]; then
      touch "${APP_DIR}/${file}"
    fi
    chmod 644 "${APP_DIR}/${file}"
  done
}

wait_for_postgres() {
  if [ -z "${LANGGRAPH_PG:-}" ]; then
    echo "环境变量 LANGGRAPH_PG 未设置，无法连接数据库" >&2
    exit 1
  fi

  python - <<'PY'
import os
import sys
import time
from urllib.parse import urlparse

import psycopg

dsn = os.environ.get("LANGGRAPH_PG")
assert dsn, "LANGGRAPH_PG 必须设置"
info = urlparse(dsn)
assert info.scheme.startswith("postgres"), "LANGGRAPH_PG 必须为 postgres URL"

max_attempts = 30
for attempt in range(1, max_attempts + 1):
    try:
        with psycopg.connect(dsn, connect_timeout=2) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            break
    except Exception as exc:
        if attempt == max_attempts:
            raise
        time.sleep(2)

# 确保 pgvector 可用
with psycopg.connect(dsn, autocommit=True) as conn:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
PY
}

main() {
  load_env_file
  create_dirs
  create_files
  wait_for_postgres

  # 默认运行 QQ 群机器人；如传入参数则使用传入命令
  if [ "$#" -gt 0 ]; then
    exec "$@"
  else
    exec python qq_group_bot.py
  fi
}

main "$@"
