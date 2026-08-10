"""记录模型 API 返回的 token 使用量。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult


class TokenUsageLogger(BaseCallbackHandler):
    """将模型成功调用的 token 使用量追加到 JSONL 文件。

    Args:
        log_path (Path | None): JSONL 日志路径；为空时使用项目 logs 目录。

    Returns:
        None: 类初始化不返回额外值。

    Raises:
        None: 初始化不主动抛出异常。
    """

    def __init__(self, log_path: Path | None = None) -> None:
        """初始化 token 使用量记录器。

        Args:
            log_path (Path | None): JSONL 日志路径；为空时使用项目 logs 目录。

        Returns:
            None: 初始化不返回额外值。

        Raises:
            None: 初始化不主动抛出异常。
        """
        self._log_path = log_path or (
            Path(__file__).resolve().parents[1] / "logs" / "token_usage.jsonl"
        )
        self._lock = Lock()

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """在模型调用成功结束后记录 token 使用量。

        Args:
            response (LLMResult): LangChain 模型调用结果。
            run_id (UUID): 当前模型调用标识。
            parent_run_id (UUID | None): 父调用标识。
            **kwargs (Any): LangChain 提供的其他回调参数。

        Returns:
            None: 记录完成后不返回额外值。

        Raises:
            None: 记录失败只输出控制台日志。
        """
        try:
            message = response.generations[0][0].message
            assert isinstance(message, AIMessage), "模型结果不是 AIMessage"
            usage = message.usage_metadata or {}
            input_details = usage.get("input_token_details") or {}
            output_details = usage.get("output_token_details") or {}
            record = {
                "time": datetime.now().astimezone().isoformat(),
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "cache_read": input_details.get("cache_read", 0),
                "reasoning": output_details.get("reasoning", 0),
            }
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            with self._lock:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                with self._log_path.open("a", encoding="utf-8") as file:
                    file.write(line + "\n")
        except Exception as exc:
            print(f"[TokenUsageLog] token 使用记录失败：{exc}", flush=True)


TOKEN_USAGE_LOGGER = TokenUsageLogger()
