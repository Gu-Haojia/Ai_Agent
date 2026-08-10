"""token 使用量 JSONL 记录测试。"""

from __future__ import annotations

import json
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
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cache_read",
        "reasoning",
    }
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
