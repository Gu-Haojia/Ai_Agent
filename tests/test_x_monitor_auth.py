"""验证 XMonitor 翻译器的后端选择与实际请求认证方式。"""

from __future__ import annotations

import os
from unittest import mock

import httpx
import pytest
from google.auth.exceptions import DefaultCredentialsError
from google.oauth2.credentials import Credentials

from src.x_monitor_translate import GeminiTweetTranslator


@pytest.mark.parametrize(
    ("vertex_flag", "key_name", "expect_vertex"),
    [
        ("true", "GEMINI_API_KEY", True),
        ("true", "GOOGLE_API_KEY", True),
        ("true", "GOOGLE_GENERATIVE_AI_API_KEY", True),
        ("true", None, True),
        ("1", "GEMINI_API_KEY", True),
        ("yes", "GEMINI_API_KEY", True),
        ("on", "GEMINI_API_KEY", True),
        (" TRUE ", "GEMINI_API_KEY", True),
        ("false", "GEMINI_API_KEY", False),
        (None, "GOOGLE_API_KEY", False),
        ("false", "GOOGLE_GENERATIVE_AI_API_KEY", False),
    ],
)
def test_translation_request_uses_selected_authentication(
    vertex_flag: str | None, key_name: str | None, expect_vertex: bool
) -> None:
    """通过真实 SDK 验证翻译请求的目标地址与认证头。

    Args:
        vertex_flag (str | None): Vertex 开关；None 表示未配置。
        key_name (str | None): API Key 环境变量名；None 表示不配置 Key。
        expect_vertex (bool): 是否应使用 Vertex 和 OAuth 凭据。

    Returns:
        None: 无返回值。

    Raises:
        AssertionError: 当路由、认证或翻译结果不符合预期时抛出。
    """
    environment = {
        "GOOGLE_CLOUD_PROJECT": "test-project",
        "GOOGLE_CLOUD_LOCATION": "global",
        "GOOGLE_APPLICATION_CREDENTIALS": "/test/service-account.json",
    }
    if vertex_flag is not None:
        environment["GOOGLE_GENAI_USE_VERTEXAI"] = vertex_flag
    if key_name is not None:
        environment[key_name] = "test-api-key"
    handler = mock.Mock(
        return_value=httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"text": '{"translations":["你好"]}'}]}}
                ]
            },
        )
    )
    with (
        mock.patch.dict(os.environ, environment, clear=True),
        mock.patch(
            "google.genai._api_client.load_auth",
            return_value=(Credentials(token="test-oauth-token"), "test-project"),
        ) as load_auth,
        mock.patch("src.x_monitor_translate.TOKEN_USAGE_LOGGER.record_google_response"),
        httpx.Client(transport=httpx.MockTransport(handler)) as transport,
        mock.patch("google.genai._api_client.SyncHttpxClient", return_value=transport),
    ):
        result = GeminiTweetTranslator(model="test-model").translate_texts(["hello"])

    assert result == ["你好"]
    handler.assert_called_once()
    request = handler.call_args.args[0]
    if expect_vertex:
        assert request.url.host == "aiplatform.googleapis.com"
        assert request.url.path == (
            "/v1beta1/projects/test-project/locations/global/"
            "publishers/google/models/test-model:generateContent"
        )
        assert request.headers["authorization"] == "Bearer test-oauth-token"
        assert "x-goog-api-key" not in request.headers
        load_auth.assert_called_once_with(project="test-project")
    else:
        assert request.url.host == "generativelanguage.googleapis.com"
        assert request.url.path == "/v1beta/models/test-model:generateContent"
        assert request.headers["x-goog-api-key"] == "test-api-key"
        assert "authorization" not in request.headers
        load_auth.assert_not_called()


@pytest.mark.parametrize("missing_name", ["GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"])
def test_vertex_requires_project_and_location_even_with_api_key(missing_name: str) -> None:
    """Vertex 配置不完整时应明确报错，即使同时提供了 API Key。

    Args:
        missing_name (str): 本次缺少的 Vertex 环境变量名。

    Returns:
        None: 无返回值。

    Raises:
        AssertionError: 当缺少必要配置仍创建客户端时抛出。
    """
    environment = {
        "GOOGLE_GENAI_USE_VERTEXAI": "true",
        "GOOGLE_CLOUD_PROJECT": "test-project",
        "GOOGLE_CLOUD_LOCATION": "global",
        "GEMINI_API_KEY": "test-api-key",
    }
    environment[missing_name] = " "
    with (
        mock.patch.dict(os.environ, environment, clear=True),
        mock.patch("google.genai.Client") as create_client,
        pytest.raises(AssertionError, match="Vertex AI.*项目与区域"),
    ):
        GeminiTweetTranslator()._create_client()
    create_client.assert_not_called()


def test_developer_api_requires_api_key() -> None:
    """未启用 Vertex 且缺少 Key 时应明确拒绝创建客户端。

    Returns:
        None: 无返回值。

    Raises:
        AssertionError: 当缺少凭据仍创建客户端时抛出。
    """
    with (
        mock.patch.dict(os.environ, {}, clear=True),
        mock.patch("google.genai.Client") as create_client,
        pytest.raises(AssertionError, match="缺少 Gemini 可用环境变量"),
    ):
        GeminiTweetTranslator()._create_client()
    create_client.assert_not_called()


def test_vertex_credential_failure_does_not_switch_to_api_key() -> None:
    """Vertex 凭据加载失败时保留原始错误，不改用同时配置的 Key。

    Returns:
        None: 无返回值。

    Raises:
        AssertionError: 当凭据错误被隐藏或仍发送请求时抛出。
    """
    environment = {
        "GOOGLE_GENAI_USE_VERTEXAI": "true",
        "GOOGLE_CLOUD_PROJECT": "test-project",
        "GOOGLE_CLOUD_LOCATION": "global",
        "GEMINI_API_KEY": "test-api-key",
    }
    with (
        mock.patch.dict(os.environ, environment, clear=True),
        mock.patch(
            "google.genai._api_client.load_auth",
            side_effect=DefaultCredentialsError("test credentials unavailable"),
        ),
        mock.patch(
            "httpx.Client.request",
            side_effect=AssertionError("凭据加载失败时不应发送请求"),
        ) as request,
        pytest.raises(DefaultCredentialsError, match="test credentials unavailable"),
    ):
        GeminiTweetTranslator(model="test-model").translate_texts(["hello"])
    request.assert_not_called()
