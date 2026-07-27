"""
xAI 环境校验测试。
"""

from __future__ import annotations

import pytest

import sql_agent_cli_stream_plus as target


@pytest.fixture(autouse=True)
def reset_xai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    重置 xAI 环境变量与模块级缓存状态。

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 提供的环境变量补丁工具。

    Returns:
        None: 无返回值。

    Raises:
        None: 本夹具不主动抛出异常。
    """
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr(target, "_ENV_XAI_CHECKED", False)


def test_ensure_xai_env_once_accepts_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    验证 XAI_API_KEY 存在时可通过校验。

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 提供的环境变量补丁工具。

    Returns:
        None: 无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    monkeypatch.setenv("XAI_API_KEY", "test-key")

    target._ensure_xai_env_once()

    assert target._ENV_XAI_CHECKED is True


def test_ensure_xai_env_once_rejects_missing_api_key() -> None:
    """
    验证缺少 XAI_API_KEY 时会抛出断言错误。

    Returns:
        None: 无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    with pytest.raises(AssertionError, match="缺少 XAI_API_KEY 环境变量"):
        target._ensure_xai_env_once()
