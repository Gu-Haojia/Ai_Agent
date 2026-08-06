"""
Gemini 环境校验测试。
"""

from __future__ import annotations

from unittest import mock

import pytest
from google.genai.errors import ClientError
from google.genai.types import HttpRetryOptions
from tenacity import Retrying, wait_none

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


def test_google_sdk_retry_codes_include_cancelled() -> None:
    """
    验证 Google SDK 的统一重试集合包含 HTTP 499。

    Returns:
        None: 无返回值。

    Raises:
        None: 预期行为由断言验证。
    """
    assert target.google_genai_api_client._RETRY_HTTP_STATUS_CODES == (
        408,
        429,
        499,
        500,
        502,
        503,
        504,
    )


def test_google_sdk_retries_cancelled_only_once() -> None:
    """
    验证 Google SDK 遇到 HTTP 499 时仅退避重试一次。

    Returns:
        None: 无返回值。

    Raises:
        None: 预期行为由断言验证。
    """
    retry_options = HttpRetryOptions(attempts=target.GOOGLE_LLM_MAX_ATTEMPTS)
    retry_kwargs = target.google_genai_api_client.retry_args(retry_options)
    retry_kwargs["wait"] = wait_none()
    retrying = Retrying(**retry_kwargs)
    operation = mock.Mock(
        side_effect=ClientError(
            499,
            {"message": "request cancelled", "status": "CANCELLED"},
        )
    )

    with pytest.raises(ClientError):
        retrying(operation)

    assert operation.call_count == 2


def test_init_google_chat_model_limits_total_attempts_to_two() -> None:
    """
    验证 Google 模型仅执行首次请求与一次退避重试。

    Returns:
        None: 无返回值。

    Raises:
        None: 预期行为由断言验证。
    """
    fake_model = object()
    with mock.patch.object(
        target,
        "init_chat_model",
        return_value=fake_model,
    ) as init_model:
        result = target._init_chat_model_with_retry(
            "google_genai:gemini-test",
            max_tokens=512,
        )

    assert result is fake_model
    init_model.assert_called_once_with(
        "google_genai:gemini-test",
        max_tokens=512,
        max_retries=2,
    )


def test_init_non_google_chat_model_keeps_provider_defaults() -> None:
    """
    验证非 Google 模型不会继承 Gemini 的重试次数语义。

    Returns:
        None: 无返回值。

    Raises:
        None: 预期行为由断言验证。
    """
    fake_model = object()
    with mock.patch.object(
        target,
        "init_chat_model",
        return_value=fake_model,
    ) as init_model:
        result = target._init_chat_model_with_retry("openai:gpt-test")

    assert result is fake_model
    init_model.assert_called_once_with("openai:gpt-test")
