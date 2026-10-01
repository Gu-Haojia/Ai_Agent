"""TTS 命令参数、HTTP 契约与群语音发送测试。"""

from __future__ import annotations

import base64
import io
import json
import wave
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from threading import Event
from unittest import mock

import httpx
import pytest

import qq_group_bot
import src.tts_command as tts_module
from qq_group_bot import BotConfig, QQBotHandler
from src.tts_command import TTSClient, TTSCommandHandler


def _wav_bytes() -> bytes:
    """构造用于协议测试的标准 PCM WAV。

    Returns:
        bytes: 包含采样数据的合法 WAV。

    Raises:
        None: 本函数不主动抛出异常。
    """
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(b"\x01\x00" * 24)
    return output.getvalue()


WAV_BYTES = _wav_bytes()


def _response(
    status_code: int = 200,
    content: bytes = WAV_BYTES,
    content_type: str = "audio/wav",
) -> httpx.Response:
    """构造具有请求上下文的 TTS HTTP 响应。

    Args:
        status_code (int): HTTP 状态码。
        content (bytes): 响应内容。
        content_type (str): 响应 MIME 类型。

    Returns:
        httpx.Response: 可校验 HTTP 状态的响应。

    Raises:
        None: 本函数不主动抛出异常。
    """
    return httpx.Response(
        status_code,
        content=content,
        headers={"Content-Type": content_type},
        request=httpx.Request("POST", "http://tts/v1/audio/speech"),
    )


def _handler(command: str = '/tts "こんにちは、世界。"') -> QQBotHandler:
    """构造可处理真实 OneBot 群消息事件的最小 Handler。

    Args:
        command (str): 测试群消息中的命令。

    Returns:
        QQBotHandler: 不启动 HTTP 服务的群消息处理器。

    Raises:
        None: 本函数不主动抛出异常。
    """
    handler = object.__new__(QQBotHandler)
    handler.bot_cfg = BotConfig(api_base="http://onebot", access_token="test-token")
    handler.agent = mock.Mock()
    handler.headers = {}
    event = {
        "post_type": "message",
        "message_type": "group",
        "self_id": 30003,
        "group_id": 10001,
        "user_id": 20002,
        "message": [
            {"type": "at", "data": {"qq": "30003"}},
            {"type": "text", "data": {"text": command}},
        ],
    }
    handler._read_body = mock.Mock(
        return_value=(json.dumps(event).encode("utf-8"), None)
    )
    handler._send_no_content = mock.Mock()
    return handler


@pytest.mark.parametrize("base_url", [None, "", "   "])
def test_unconfigured_command_is_silent(base_url: str | None) -> None:
    """验证没有有效环境变量时不调用服务、不发消息、不进入 Agent。

    Args:
        base_url (str | None): 缺失、空或仅含空白的配置。

    Returns:
        None: 无返回值。

    Raises:
        None: 预期行为由断言验证。
    """
    env = {} if base_url is None else {"TTS_API_BASE": base_url}
    handler = _handler()
    with (
        mock.patch.dict(tts_module.os.environ, env, clear=True),
        mock.patch.object(tts_module, "_TTS_EXECUTOR") as executor,
        mock.patch.object(tts_module.httpx, "post") as http_post,
        mock.patch.object(qq_group_bot, "_call_onebot_action") as send_action,
        mock.patch.object(qq_group_bot, "_send_group_msg") as send_text,
    ):
        assert TTSClient.from_env() is None
        handler.do_POST()
    handler._send_no_content.assert_called_once_with()
    handler.agent.chat_once_stream.assert_not_called()
    executor.submit.assert_not_called()
    http_post.assert_not_called()
    send_action.assert_not_called()
    send_text.assert_not_called()


@pytest.mark.parametrize(
    "base_url",
    ["192.168.3.221:18763", "ftp://tts", "http://tts?voice=x", "http://["],
)
def test_invalid_service_address_is_explicit(base_url: str) -> None:
    """验证地址不合法时明确抛出断言。

    Args:
        base_url (str): 非法服务基地址。

    Returns:
        None: 无返回值。

    Raises:
        None: 被测断言由 pytest 捕获。
    """
    with pytest.raises(AssertionError, match="TTS_API_BASE"):
        TTSClient(base_url)


def test_client_preserves_text_and_uses_documented_endpoint() -> None:
    """验证真实 HTTP 请求编码保留原文并沿用服务端默认模型和角色。

    Returns:
        None: 无返回值。

    Raises:
        None: 预期行为由断言验证。
    """
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        """记录发送的 HTTP 请求并返回 WAV。

        Args:
            request (httpx.Request): 编码完成的请求。

        Returns:
            httpx.Response: TTS 成功响应。

        Raises:
            None: 本函数不主动抛出异常。
        """
        requests.append(request)
        return _response()

    text = '  こんにちは、\n"プロデューサーさん"。  '
    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        client = TTSClient("http://tts/prefix/", http_post=http_client.post)
        assert client.synthesize(text) == WAV_BYTES
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == "http://tts/prefix/v1/audio/speech"
    assert json.loads(request.content) == {"input": text, "response_format": "wav"}
    assert request.extensions["timeout"]["read"] == 300.0


@pytest.mark.parametrize(
    ("response", "error_type"),
    [
        (_response(status_code=503), httpx.HTTPStatusError),
        (_response(status_code=202), AssertionError),
        (_response(content_type="application/json"), AssertionError),
        (_response(content=b"invalid"), AssertionError),
        (_response(content=b""), AssertionError),
    ],
)
def test_invalid_upstream_response_never_becomes_audio(
    response: httpx.Response,
    error_type: type[Exception],
) -> None:
    """验证服务失败、错误状态或非 WAV 响应会显式失败。

    Args:
        response (httpx.Response): 非法上游响应。
        error_type (type[Exception]): 预期异常类型。

    Returns:
        None: 无返回值。

    Raises:
        None: 被测异常由 pytest 捕获。
    """
    client = TTSClient("http://tts", http_post=mock.Mock(return_value=response))
    with pytest.raises(error_type):
        client.synthesize("こんにちは")


@pytest.mark.parametrize(
    "command",
    [
        "/tts",
        "/tts hello",
        '/tts ""',
        '/tts "   "',
        '/tts "hello',
        '/tts "hello" "world"',
        '/tts "hello" extra',
        '/tts "' + "あ" * 2001 + '"',
    ],
)
def test_invalid_command_is_rejected_before_scheduling(command: str) -> None:
    """验证参数错误时向原群提示错误并且不提交生成任务。

    Args:
        command (str): 非法语音命令。

    Returns:
        None: 无返回值。

    Raises:
        None: 预期行为由断言验证。
    """
    executor = mock.Mock(spec=Executor)
    send_action = mock.Mock()
    with mock.patch.dict(tts_module.os.environ, {"TTS_API_BASE": "http://tts"}):
        TTSCommandHandler(send_action, executor).handle(command, 10001, "http://onebot")
    executor.submit.assert_not_called()
    payload = send_action.call_args.args[2]
    assert payload["group_id"] == 10001
    assert payload["message"].startswith("TTS 参数错误：")


@pytest.mark.parametrize(
    ("command", "text"),
    [
        ('/tts "こんにちは、世界。"', "こんにちは、世界。"),
        ('/tts "hello world\nnext line"', "hello world\nnext line"),
        ('/tts "say \\"hello\\""', 'say "hello"'),
        ('/tts "' + "あ" * 2000 + '"', "あ" * 2000),
    ],
)
def test_quoted_text_is_submitted_without_waiting(command: str, text: str) -> None:
    """验证带空格、换行和转义引号的原文进入后台任务。

    Args:
        command (str): 合法的完整命令。
        text (str): 预期合成文本。

    Returns:
        None: 无返回值。

    Raises:
        None: 预期行为由断言验证。
    """
    pending: Future[None] = Future()
    executor = mock.Mock(spec=Executor)
    executor.submit.return_value = pending
    send_action = mock.Mock()
    with mock.patch.dict(tts_module.os.environ, {"TTS_API_BASE": "http://tts"}):
        TTSCommandHandler(send_action, executor).handle(command, 10001, "http://onebot")
    assert executor.submit.call_args.args[2] == text
    assert pending.done() is False
    send_action.assert_not_called()


def test_group_callback_acknowledges_before_audio_and_sends_record() -> None:
    """验证慢速生成不持有群消息锁，且经真实 OneBot 调用构造语音消息。

    Returns:
        None: 无返回值。

    Raises:
        None: 预期行为由断言验证。
    """
    started = Event()
    release = Event()
    http_calls: list[tuple[str, dict[str, str]]] = []

    def slow_post(
        url: str, *, json: dict[str, str], timeout: httpx.Timeout
    ) -> httpx.Response:
        """等待主线程放行，模拟耗时的 TTS 请求。

        Args:
            url (str): 请求地址。
            json (dict[str, str]): 合成参数。
            timeout (httpx.Timeout): 请求超时配置。

        Returns:
            httpx.Response: WAV 响应。

        Raises:
            AssertionError: 当命令处理线程未及时返回时抛出。
        """
        http_calls.append((url, json))
        started.set()
        assert release.wait(5), "群消息处理被语音生成阻塞"
        return _response()

    onebot_response = mock.MagicMock(status=200)
    onebot_response.read.return_value = b'{"status":"ok","retcode":0}'
    onebot_response.__enter__.return_value = onebot_response
    handler = _handler()
    with (
        ThreadPoolExecutor(max_workers=1) as executor,
        mock.patch.object(tts_module, "_TTS_EXECUTOR", executor),
        mock.patch.dict(tts_module.os.environ, {"TTS_API_BASE": "http://tts"}),
        mock.patch.object(tts_module.httpx, "post", side_effect=slow_post),
        mock.patch.object(qq_group_bot, "urlopen", return_value=onebot_response) as send,
        mock.patch.object(qq_group_bot, "_send_group_msg") as send_text,
        mock.patch.object(QQBotHandler, "power_enabled", False),
    ):
        try:
            handler.do_POST()
            assert started.wait(2)
            handler._send_no_content.assert_called_once_with()
            handler.agent.chat_once_stream.assert_not_called()
            send.assert_not_called()
            assert _handler()._handle_commands(10001, 20002, "/cmd") is True
            assert '/tts "文本"' in send_text.call_args.args[2]
        finally:
            release.set()
        executor.shutdown(wait=True)
    assert http_calls == [
        (
            "http://tts/v1/audio/speech",
            {"input": "こんにちは、世界。", "response_format": "wav"},
        )
    ]
    send.assert_called_once()
    request = send.call_args.args[0]
    assert request.full_url == "http://onebot/send_group_msg"
    assert request.get_header("Authorization") == "Bearer test-token"
    payload = json.loads(request.data)
    assert payload["group_id"] == 10001
    assert len(payload["message"]) == 1
    record = payload["message"][0]
    assert record["type"] == "record"
    audio_file = record["data"]["file"]
    assert audio_file.startswith("base64://")
    assert base64.b64decode(audio_file[len("base64://") :]) == WAV_BYTES


@pytest.mark.parametrize(
    "error",
    [httpx.ReadTimeout("request timed out"), AssertionError("无有效 WAV")],
)
def test_generation_failure_returns_text_to_original_group(error: Exception) -> None:
    """验证生成失败时不发送语音，而向原群明确报告错误。

    Args:
        error (Exception): 生成过程的预期异常。

    Returns:
        None: 无返回值。

    Raises:
        None: 预期行为由断言验证。
    """
    client = mock.Mock(spec=TTSClient)
    client.synthesize.side_effect = error
    send_action = mock.Mock()
    handler = TTSCommandHandler(send_action)
    handler._generate_and_send(client, "hello", 10001, "http://onebot", "test-token")
    send_action.assert_called_once_with(
        "http://onebot",
        "send_group_msg",
        {"group_id": 10001, "message": f"TTS 生成失败：{error}"},
        "test-token",
    )


def test_napcat_business_failure_is_visible(capsys: pytest.CaptureFixture[str]) -> None:
    """验证 NapCat 返回 HTTP 200 但业务失败时会留下明确错误日志。

    Args:
        capsys (pytest.CaptureFixture[str]): 控制台输出捕获器。

    Returns:
        None: 无返回值。

    Raises:
        None: 预期行为由断言验证。
    """
    response = mock.MagicMock(status=200)
    response.read.return_value = b'{"status":"failed","retcode":1200,"message":"denied"}'
    response.__enter__.return_value = response
    with (
        ThreadPoolExecutor(max_workers=1) as executor,
        mock.patch.object(tts_module, "_TTS_EXECUTOR", executor),
        mock.patch.dict(tts_module.os.environ, {"TTS_API_BASE": "http://tts"}),
        mock.patch.object(tts_module.httpx, "post", return_value=_response()),
        mock.patch.object(qq_group_bot, "urlopen", return_value=response),
    ):
        assert _handler()._handle_commands(10001, 20002, '/tts "hello"') is True
        executor.shutdown(wait=True)
    output = capsys.readouterr().err
    assert "[TTS] 语音命令执行失败" in output
    assert "retcode=1200" in output


def test_tts_respects_command_whitelist() -> None:
    """验证未授权用户无法提交 TTS 任务。

    Returns:
        None: 无返回值。

    Raises:
        None: 预期行为由断言验证。
    """
    handler = _handler()
    handler.bot_cfg.cmd_allowed_users = (99999,)
    with (
        mock.patch.dict(tts_module.os.environ, {"TTS_API_BASE": "http://tts"}),
        mock.patch.object(tts_module, "_TTS_EXECUTOR") as executor,
        mock.patch.object(qq_group_bot, "_send_group_msg") as send,
    ):
        assert handler._handle_commands(10001, 20002, '/tts "hello"') is True
    executor.submit.assert_not_called()
    assert send.call_args.args[2] == "无权执行命令（需在白名单内）。"


def test_disabled_command_list_and_unknown_commands_are_unchanged() -> None:
    """验证禁用时命令列表没有 TTS，近似命令仍按原流程处理。

    Returns:
        None: 无返回值。

    Raises:
        None: 预期行为由断言验证。
    """
    with (
        mock.patch.dict(tts_module.os.environ, {"TTS_API_BASE": ""}),
        mock.patch.object(qq_group_bot, "_send_group_msg") as send,
    ):
        handler = _handler()
        assert handler._handle_commands(10001, 20002, "/cmd") is True
        assert "/tts" not in send.call_args.args[2]
        assert handler._handle_commands(10001, 20002, '/ttsx "hello"') is True
        assert send.call_args.args[2] == "无此命令"
