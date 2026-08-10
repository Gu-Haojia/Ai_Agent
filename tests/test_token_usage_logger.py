"""token 使用量 JSONL 记录测试。"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from src.token_usage_logger import TokenUsageLogger


def test_token_usage_logger_writes_flat_usage_fields(tmp_path: Path) -> None:
    """验证记录器写入完整的扁平 token 字段。

    Args:
        tmp_path (Path): pytest 临时目录。

    Returns:
        None: 测试完成后不返回额外值。

    Raises:
        None: 预期行为由断言验证。
    """
    log_path = tmp_path / "token_usage.jsonl"
    logger = TokenUsageLogger(log_path)
    message = AIMessage(
        content="完成",
        response_metadata={"model_name": "gemini-3.6-flash"},
        usage_metadata={
            "input_tokens": 120,
            "output_tokens": 30,
            "total_tokens": 150,
            "input_token_details": {"cache_read": 80},
            "output_token_details": {"reasoning": 10},
        },
    )

    logger.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=message)]]),
        run_id=uuid4(),
    )

    record = json.loads(log_path.read_text(encoding="utf-8"))
    assert set(record) == {
        "time",
        "model_name",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cache_read",
        "reasoning",
    }
    assert record["model_name"] == "gemini-3.6-flash"
    assert record["input_tokens"] == 120
    assert record["output_tokens"] == 30
    assert record["total_tokens"] == 150
    assert record["cache_read"] == 80
    assert record["reasoning"] == 10


def test_token_usage_logger_fills_missing_usage_with_zero(tmp_path: Path) -> None:
    """验证 token 元数据缺失时所有数值字段补零。

    Args:
        tmp_path (Path): pytest 临时目录。

    Returns:
        None: 测试完成后不返回额外值。

    Raises:
        None: 预期行为由断言验证。
    """
    log_path = tmp_path / "token_usage.jsonl"
    logger = TokenUsageLogger(log_path)
    message = AIMessage(content="完成")

    logger.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=message)]]),
        run_id=uuid4(),
    )

    record = json.loads(log_path.read_text(encoding="utf-8"))
    assert record["input_tokens"] == 0
    assert record["model_name"] == ""
    assert record["output_tokens"] == 0
    assert record["total_tokens"] == 0
    assert record["cache_read"] == 0
    assert record["reasoning"] == 0


def test_token_usage_logger_only_prints_when_recording_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """验证记录失败时只输出控制台日志。

    Args:
        tmp_path (Path): pytest 临时目录。
        capsys (pytest.CaptureFixture[str]): pytest 控制台输出捕获器。

    Returns:
        None: 测试完成后不返回额外值。

    Raises:
        None: 预期行为由断言验证。
    """
    logger = TokenUsageLogger(tmp_path / "token_usage.jsonl")

    logger.on_llm_end(LLMResult(generations=[]), run_id=uuid4())

    captured = capsys.readouterr()
    assert "[TokenUsageLog] token 使用记录失败：" in captured.out


def test_token_usage_logger_summarizes_from_selected_time(tmp_path: Path) -> None:
    """验证记录器只汇总指定时间及之后的记录。

    Args:
        tmp_path (Path): pytest 临时目录。

    Returns:
        None: 测试完成后不返回额外值。

    Raises:
        None: 预期行为由断言验证。
    """
    log_path = tmp_path / "token_usage.jsonl"
    records = [
        {
            "time": "2026-08-10T10:00:00+09:00",
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "cache_read": 40,
            "reasoning": 5,
        },
        {
            "time": "2026-08-10T12:00:00+09:00",
            "input_tokens": 200,
            "output_tokens": 30,
            "total_tokens": 230,
            "cache_read": 80,
            "reasoning": 10,
        },
    ]
    log_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    logger = TokenUsageLogger(log_path)

    summary = logger.summarize(
        datetime(2026, 8, 10, 11, 0, tzinfo=timezone(timedelta(hours=9)))
    )

    assert summary.start_time == datetime.fromisoformat(records[1]["time"])
    assert summary.total_tokens == 230
    assert summary.input_tokens == 200
    assert summary.cache_read == 80
    assert summary.output_tokens == 30


def test_token_usage_logger_clear_keeps_empty_file(tmp_path: Path) -> None:
    """验证清空操作保留空日志文件。

    Args:
        tmp_path (Path): pytest 临时目录。

    Returns:
        None: 测试完成后不返回额外值。

    Raises:
        None: 预期行为由断言验证。
    """
    log_path = tmp_path / "token_usage.jsonl"
    log_path.write_text('{"time":"2026-08-10T10:00:00+09:00"}\n', encoding="utf-8")
    logger = TokenUsageLogger(log_path)

    logger.clear()

    assert log_path.exists()
    assert log_path.read_text(encoding="utf-8") == ""
