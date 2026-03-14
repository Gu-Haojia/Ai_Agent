"""
Gemini 环境校验测试。
"""

from __future__ import annotations

import pytest

import sql_agent_cli_stream_plus as target


_CREDENTIAL_ENV_NAMES: tuple[str, ...] = (
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_GENERATIVE_AI_API_KEY",
    "GOOGLE_GENAI_USE_VERTEXAI",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
    "GOOGLE_APPLICATION_CREDENTIALS",
)


@pytest.fixture(autouse=True)
def reset_gemini_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    重置 Gemini 相关环境变量与模块级缓存状态。

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 提供的环境变量补丁工具。

    Returns:
        None: 无返回值。

    Raises:
        None: 本夹具不主动抛出异常。
    """
    for env_name in _CREDENTIAL_ENV_NAMES:
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setattr(target, "_ENV_GEMINI_CHECKED", False)


@pytest.mark.parametrize(
    "env_name",
    ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY"),
)
def test_ensure_gemini_env_once_accepts_ai_studio_key(
    monkeypatch: pytest.MonkeyPatch, env_name: str
) -> None:
    """
    验证 AI Studio 任一兼容密钥都能通过校验。

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 提供的环境变量补丁工具。
        env_name (str): 本次测试使用的环境变量名。

    Returns:
        None: 无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    monkeypatch.setenv(env_name, "test-key")

    target._ensure_gemini_env_once()

    assert target._ENV_GEMINI_CHECKED is True


def test_ensure_gemini_env_once_accepts_vertex_env_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    验证 Vertex 相关环境变量存在时可通过校验。

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 提供的环境变量补丁工具。

    Returns:
        None: 无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/app/vertex-sa.json")

    target._ensure_gemini_env_once()

    assert target._ENV_GEMINI_CHECKED is True


def test_ensure_gemini_env_once_rejects_missing_credentials() -> None:
    """
    验证缺少 AI Studio 与 Vertex 环境变量时会抛出断言错误。

    Returns:
        None: 无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    with pytest.raises(AssertionError, match="缺少 Gemini 可用环境变量"):
        target._ensure_gemini_env_once()
