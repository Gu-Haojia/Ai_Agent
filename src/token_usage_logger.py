"""记录模型 API 返回的 token 使用量。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult


@dataclass(frozen=True)
class TokenUsageSummary:
    """token 使用量汇总结果。

    Args:
        start_time (datetime | None): 汇总记录中的最早时间。
        total_tokens (int): 总 token 数。
        input_tokens (int): 输入 token 数。
        cache_read (int): 缓存命中 token 数。
        output_tokens (int): 输出 token 数。

    Returns:
        None: 数据类初始化不返回额外值。

    Raises:
        None: 数据类初始化不主动抛出异常。
    """

    start_time: datetime | None
    total_tokens: int
    input_tokens: int
    cache_read: int
    output_tokens: int


@dataclass(frozen=True)
class TokenUsageRecord:
    """单次模型调用的 Token 使用记录。

    Args:
        recorded_at (datetime): 调用完成时间。
        model_name (str): API 返回的模型名称。
        total_tokens (int): 总 Token 数。
        input_tokens (int): 输入 Token 数。
        cache_read (int): 缓存命中 Token 数。
        output_tokens (int): 输出 Token 数。

    Returns:
        None: 数据类初始化不返回额外值。

    Raises:
        None: 数据类初始化不主动抛出异常。
    """

    recorded_at: datetime
    model_name: str
    total_tokens: int
    input_tokens: int
    cache_read: int
    output_tokens: int


@dataclass(frozen=True)
class TokenUsageReport:
    """同一日志快照中的汇总结果与明细记录。

    Args:
        summary (TokenUsageSummary): Token 汇总结果。
        records (tuple[TokenUsageRecord, ...]): 符合筛选条件的调用记录。
        end_time (datetime | None): 汇总记录中的最晚时间。

    Returns:
        None: 数据类初始化不返回额外值。

    Raises:
        None: 数据类初始化不主动抛出异常。
    """

    summary: TokenUsageSummary
    records: tuple[TokenUsageRecord, ...]
    end_time: datetime | None


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
            self.record_usage(
                model_name=str(message.response_metadata.get("model_name") or ""),
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                total_tokens=usage.get("total_tokens"),
                cache_read=input_details.get("cache_read"),
                reasoning=output_details.get("reasoning"),
            )
        except Exception as exc:
            print(f"[TokenUsageLog] token 使用记录失败：{exc}", flush=True)

    def record_google_response(self, response: Any, requested_model: str) -> None:
        """解析 Google GenAI 响应并记录 token 使用量。

        Args:
            response (Any): Google GenAI 的生成响应。
            requested_model (str): 请求使用的模型名称。

        Returns:
            None: 记录完成后不返回额外值。

        Raises:
            None: 解析或记录失败只输出控制台日志。
        """
        try:
            usage = getattr(response, "usage_metadata", None)
            self.record_usage(
                model_name=str(
                    getattr(response, "model_version", None) or requested_model
                ),
                input_tokens=getattr(usage, "prompt_token_count", 0),
                output_tokens=getattr(usage, "candidates_token_count", 0),
                total_tokens=getattr(usage, "total_token_count", 0),
                cache_read=getattr(usage, "cached_content_token_count", 0),
                reasoning=getattr(usage, "thoughts_token_count", 0),
            )
        except Exception as exc:
            print(f"[TokenUsageLog] token 使用记录失败：{exc}", flush=True)

    def record_usage(
        self,
        *,
        model_name: str,
        input_tokens: int | None,
        output_tokens: int | None,
        total_tokens: int | None,
        cache_read: int | None,
        reasoning: int | None,
    ) -> None:
        """将标准化 token 使用量追加到 JSONL 文件。

        Args:
            model_name (str): API 返回或请求使用的模型名称。
            input_tokens (int | None): 输入 token 数。
            output_tokens (int | None): 输出 token 数。
            total_tokens (int | None): 总 token 数。
            cache_read (int | None): 缓存命中 token 数。
            reasoning (int | None): 推理 token 数。

        Returns:
            None: 记录完成后不返回额外值。

        Raises:
            None: 写入失败只输出控制台日志。
        """
        try:
            record = {
                "time": datetime.now().astimezone().isoformat(),
                "model_name": model_name,
                "input_tokens": input_tokens or 0,
                "output_tokens": output_tokens or 0,
                "total_tokens": total_tokens or 0,
                "cache_read": cache_read or 0,
                "reasoning": reasoning or 0,
            }
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            with self._lock:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                with self._log_path.open("a", encoding="utf-8") as file:
                    file.write(line + "\n")
        except Exception as exc:
            print(f"[TokenUsageLog] token 使用记录失败：{exc}", flush=True)

    def summarize(self, start_time: datetime | None = None) -> TokenUsageSummary:
        """汇总指定时间之后的 token 使用量。

        Args:
            start_time (datetime | None): 筛选起始时间；为空时汇总全部记录。

        Returns:
            TokenUsageSummary: 符合时间条件的 token 汇总结果。

        Raises:
            AssertionError: 当筛选时间不含时区或日志字段非法时抛出。
            OSError: 当日志文件无法读取时抛出。
            ValueError: 当日志内容不是有效 JSON 或时间格式非法时抛出。
        """
        return self.report(start_time).summary

    def report(self, start_time: datetime | None = None) -> TokenUsageReport:
        """读取同一日志快照并返回汇总结果与明细。

        Args:
            start_time (datetime | None): 筛选起始时间；为空时读取全部记录。

        Returns:
            TokenUsageReport: 符合时间条件的日志报告。

        Raises:
            AssertionError: 当筛选时间不含时区或日志字段非法时抛出。
            OSError: 当日志文件无法读取时抛出。
            ValueError: 当日志内容不是有效 JSON 或时间格式非法时抛出。
        """
        if start_time is not None:
            assert start_time.tzinfo is not None, "筛选时间必须包含时区"
        records: list[TokenUsageRecord] = []
        totals = {
            "total_tokens": 0,
            "input_tokens": 0,
            "cache_read": 0,
            "output_tokens": 0,
        }
        with self._lock:
            if self._log_path.exists():
                with self._log_path.open("r", encoding="utf-8") as file:
                    for line in file:
                        if not line.strip():
                            continue
                        raw_record = json.loads(line)
                        recorded_at = datetime.fromisoformat(raw_record["time"])
                        assert recorded_at.tzinfo is not None, (
                            "日志时间必须包含时区"
                        )
                        if start_time is not None and recorded_at < start_time:
                            continue
                        values: dict[str, int] = {}
                        for field in totals:
                            value = raw_record[field]
                            assert isinstance(value, int), f"{field} 必须为整数"
                            values[field] = value
                            totals[field] += value
                        model_name = raw_record.get("model_name", "")
                        assert isinstance(model_name, str), "model_name 必须为字符串"
                        records.append(
                            TokenUsageRecord(
                                recorded_at=recorded_at,
                                model_name=model_name,
                                total_tokens=values["total_tokens"],
                                input_tokens=values["input_tokens"],
                                cache_read=values["cache_read"],
                                output_tokens=values["output_tokens"],
                            )
                        )
        records.sort(key=lambda record: record.recorded_at)
        earliest = records[0].recorded_at if records else None
        end_time = records[-1].recorded_at if records else None
        summary = TokenUsageSummary(
            start_time=earliest,
            total_tokens=totals["total_tokens"],
            input_tokens=totals["input_tokens"],
            cache_read=totals["cache_read"],
            output_tokens=totals["output_tokens"],
        )
        return TokenUsageReport(summary, tuple(records), end_time)

    def clear(self) -> None:
        """清空 token 使用量日志内容。

        Returns:
            None: 清空完成后不返回额外值。

        Raises:
            OSError: 当日志目录或文件无法写入时抛出。
        """
        with self._lock:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_path.write_text("", encoding="utf-8")


TOKEN_USAGE_LOGGER = TokenUsageLogger()
