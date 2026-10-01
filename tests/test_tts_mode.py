"""TTS 模式隔离、图片处理、后台投递与原流程回归测试。"""

from __future__ import annotations

import base64
import io
import json
import wave
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest import mock

import httpx
import pytest
from google.auth.exceptions import GoogleAuthError
from google.genai import errors, types

import qq_group_bot
import src.tts_mode as mode_module
from image_storage import GeneratedImage, ImageStorageManager, StoredImage
from qq_group_bot import BotConfig, QQBotHandler
from src.tts_command import TTSClient
from src.tts_mode import JapaneseReplyRewriter, TTSModeService, TTSReplyJob


@pytest.fixture(autouse=True)
def isolated_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """隔离各测试的群开关，防止影响已有 QQ 测试。

    Args:
        monkeypatch (pytest.MonkeyPatch): 自动恢复共享状态的工具。

    Returns:
        None: 无返回值。

    Raises:
        None: 本函数不主动抛出异常；测试结果由断言验证。
    """
    monkeypatch.setattr(QQBotHandler, "tts_mode", None)
    monkeypatch.setattr(QQBotHandler, "power_enabled", True)
    monkeypatch.setattr(QQBotHandler, "image_storage", None)


def _wav_bytes() -> bytes:
    """构造用于完整投递测试的 PCM WAV。

    Returns:
        bytes: 合法 WAV 音频。

    Raises:
        None: 本函数不主动抛出异常；测试结果由断言验证。
    """
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(b"\x01\x00" * 24)
    return output.getvalue()


WAV_BYTES = _wav_bytes()


def _response(text: str = "そなた、一緒に参りましょうー。") -> types.GenerateContentResponse:
    """构造 Google 原生 API 的正常文本响应。

    Args:
        text (str): 候选日语正文。

    Returns:
        types.GenerateContentResponse: 正常完成的 API 响应。

    Raises:
        None: 本函数不主动抛出异常；测试结果由断言验证。
    """
    return types.GenerateContentResponse(
        candidates=[types.Candidate(
            content=types.Content(parts=[types.Part(text=text)]),
            finish_reason=types.FinishReason.STOP,
        )]
    )


def _service() -> tuple[TTSModeService, mock.Mock, mock.Mock, mock.Mock, mock.Mock]:
    """构造可检查后台任务和消息的隔离服务。

    Returns:
        tuple: 服务、消息发送器、执行器、TTS 客户端及改写器。

    Raises:
        None: 本函数不主动抛出异常；测试结果由断言验证。
    """
    send = mock.Mock()
    executor = mock.Mock()
    executor.submit.return_value = Future()
    tts = mock.Mock(spec=TTSClient)
    tts.synthesize.return_value = WAV_BYTES
    rewriter = mock.Mock(spec=JapaneseReplyRewriter)
    rewriter.rewrite.return_value = "そなた、一緒に参りましょうー。"
    service = TTSModeService(send, QQBotHandler._compose_group_message, executor)
    return service, send, executor, tts, rewriter


def _enable(service: TTSModeService, tts: mock.Mock, rewriter: mock.Mock) -> None:
    """使用真实命令入口为测试群启用模式。

    Args:
        service (TTSModeService): 被测服务。
        tts (mock.Mock): 注入的语音客户端。
        rewriter (mock.Mock): 注入的日语改写器。

    Returns:
        None: 无返回值。

    Raises:
        None: 本函数不主动抛出异常；测试结果由断言验证。
    """
    with (
        mock.patch.object(TTSClient, "from_env", return_value=tts),
        mock.patch.object(JapaneseReplyRewriter, "from_env", return_value=rewriter),
    ):
        service.handle_command("/ttsmode on", 10001, "http://onebot", "token")


def _handler(
    answer: str = "我们一起去吧。", group_id: int = 10001, text: str = "一起去吗？"
) -> QQBotHandler:
    """构造保留真实群消息解析及回复出口的最小 Handler。

    Args:
        answer (str): 主 Agent 的测试回复。
        group_id (int): 当前群号。
        text (str): 用户消息正文或控制命令。

    Returns:
        QQBotHandler: 不启动 HTTP 服务的处理器。

    Raises:
        None: 本函数不主动抛出异常；测试结果由断言验证。
    """
    handler = object.__new__(QQBotHandler)
    handler.bot_cfg = BotConfig(api_base="http://onebot", access_token="token")
    handler.agent = SimpleNamespace(
        _config=SimpleNamespace(model_name="google_genai:gemini-test"),
        set_token_printer=mock.Mock(),
        set_memory_namespace=mock.Mock(),
        chat_once_stream=mock.Mock(return_value=answer),
        consume_generated_images=mock.Mock(return_value=[]),
    )
    handler.headers = {}
    handler._namespace_for = mock.Mock(return_value="group-memory")
    handler._thread_id_for = mock.Mock(return_value="original-thread")
    handler._send_no_content = mock.Mock()
    event = {
        "post_type": "message", "message_type": "group", "self_id": 30003,
        "group_id": group_id, "user_id": 20002,
        "message": [
            {"type": "at", "data": {"qq": "30003"}},
            {"type": "text", "data": {"text": text}},
        ],
    }
    handler._read_body = mock.Mock(return_value=(json.dumps(event).encode(), None))
    return handler


def test_rewriter_uses_only_one_round_and_closes_client() -> None:
    """验证独立调用无历史、无工具，且关闭连接。

    Returns:
        None: 无返回值。

    Raises:
        None: 本函数不主动抛出异常；测试结果由断言验证。
    """
    client = mock.Mock()
    client.models.generate_content.return_value = _response()
    rewriter = JapaneseReplyRewriter("gemini-3.8-flash", "格式和原句", lambda: client)
    assert rewriter.rewrite("我们一起去吧。") == "そなた、一緒に参りましょうー。"
    call = client.models.generate_content.call_args
    assert call.kwargs["model"] == "gemini-3.8-flash"
    assert call.kwargs["contents"] == "待改写回答：\n我们一起去吧。"
    config = call.kwargs["config"]
    assert config.system_instruction == "格式和原句"
    assert config.thinking_config.thinking_level == types.ThinkingLevel.LOW
    assert config.tools is None
    client.close.assert_called_once()


@pytest.mark.parametrize("response", [
    types.GenerateContentResponse(), _response(""),
    types.GenerateContentResponse(candidates=[types.Candidate(
        finish_reason=types.FinishReason.MAX_TOKENS,
    )]),
])
def test_invalid_generation_is_explicit(response: types.GenerateContentResponse) -> None:
    """验证空结果和截断结果明确失败，不使用原文替代。

    Args:
        response (types.GenerateContentResponse): 无效模型结果。

    Returns:
        None: 无返回值。

    Raises:
        None: 本函数不主动抛出异常；测试结果由断言验证。
    """
    client = mock.Mock()
    client.models.generate_content.return_value = response
    with pytest.raises(AssertionError):
        JapaneseReplyRewriter("gemini-test", "提示词", lambda: client).rewrite("正文")
    client.close.assert_called_once()


def test_env_configuration_is_snapshotted_and_prompt_is_separate(tmp_path: Path) -> None:
    """验证可替换模型和提示词，不复用主 Agent 的系统提示词。

    Args:
        tmp_path (Path): 临时提示词目录。

    Returns:
        None: 无返回值。

    Raises:
        None: 本函数不主动抛出异常；测试结果由断言验证。
    """
    prompt = tmp_path / "tts.txt"
    prompt.write_text("独立原句校准", encoding="utf-8")
    client = mock.Mock()
    client.models.generate_content.return_value = _response()
    with (
        mock.patch.dict(mode_module.os.environ, {
            "GEMINI_API_KEY": "test-key", "TTS_REWRITE_MODEL": "gemini-3.5-flash-lite",
            "TTS_REWRITE_PROMPT_FILE": str(prompt), "SYS_MSG_FILE": "主提示词不存在",
            "GOOGLE_GENAI_USE_VERTEXAI": "true",
        }, clear=True),
        mock.patch.object(mode_module.genai, "Client", return_value=client) as create,
    ):
        rewriter = JapaneseReplyRewriter.from_env()
        mode_module.os.environ["GEMINI_API_KEY"] = "changed-key"
        prompt.write_text("后续修改", encoding="utf-8")
        rewriter.rewrite("正文")
    assert create.call_args.kwargs["api_key"] == "test-key"
    assert create.call_args.kwargs["vertexai"] is False
    config = client.models.generate_content.call_args.kwargs
    assert config["model"] == "gemini-3.5-flash-lite"
    assert config["config"].system_instruction == "独立原句校准"


def test_vertex_configuration_and_default_prompt() -> None:
    """验证已有 Vertex AI 环境也能启用，默认模板随仓库分发。

    Returns:
        None: 无返回值。

    Raises:
        None: 本函数不主动抛出异常；测试结果由断言验证。
    """
    client = mock.Mock()
    client.models.generate_content.return_value = _response()
    with (
        mock.patch.dict(mode_module.os.environ, {
            "GOOGLE_GENAI_USE_VERTEXAI": "true", "GOOGLE_CLOUD_PROJECT": "test-project",
            "GOOGLE_CLOUD_LOCATION": "global",
        }, clear=True),
        mock.patch.object(mode_module.genai, "Client", return_value=client) as create,
    ):
        JapaneseReplyRewriter.from_env().rewrite("正文")
    assert create.call_args.kwargs["vertexai"] is True
    assert create.call_args.kwargs["project"] == "test-project"
    assert "<游戏台词>" in client.models.generate_content.call_args.kwargs["config"].system_instruction


def test_disabled_mode_has_no_configuration_or_external_calls() -> None:
    """验证默认关闭时连配置与执行器都不会访问。

    Returns:
        None: 无返回值。

    Raises:
        None: 本函数不主动抛出异常；测试结果由断言验证。
    """
    service, send, executor, _, _ = _service()
    with mock.patch.object(JapaneseReplyRewriter, "from_env") as factory:
        assert service.enqueue_if_enabled(10001, "原回复", [], None, "http://onebot", "") is False
    factory.assert_not_called()
    executor.submit.assert_not_called()
    send.assert_not_called()


def test_toggle_and_explicit_off_affect_only_current_group() -> None:
    """验证无参数开关、显式关闭及群隔离。

    Returns:
        None: 无返回值。

    Raises:
        None: 本函数不主动抛出异常；测试结果由断言验证。
    """
    service, _, executor, tts, rewriter = _service()
    _enable(service, tts, rewriter)
    assert service.enqueue_if_enabled(10002, "正文", [], None, "http://onebot", "") is False
    assert service.enqueue_if_enabled(10001, "正文", [], None, "http://onebot", "") is True
    service.handle_command("/ttsmode", 10001, "http://onebot", "")
    assert service.enqueue_if_enabled(10001, "正文", [], None, "http://onebot", "") is False
    with (
        mock.patch.object(TTSClient, "from_env", return_value=tts),
        mock.patch.object(JapaneseReplyRewriter, "from_env", return_value=rewriter),
    ):
        service.handle_command("/ttsmode", 10001, "http://onebot", "")
    service.handle_command("/ttsmode off", 10001, "http://onebot", "")
    assert service.enqueue_if_enabled(10001, "正文", [], None, "http://onebot", "") is False
    assert executor.submit.call_count == 1


@pytest.mark.parametrize("command", ["/ttsmode yes", "/ttsmode on extra"])
def test_invalid_command_does_not_enable(command: str) -> None:
    """验证格式错误不会创建配置或开启模式。

    Args:
        command (str): 非法控制命令。

    Returns:
        None: 无返回值。

    Raises:
        None: 本函数不主动抛出异常；测试结果由断言验证。
    """
    service, send, _, _, _ = _service()
    with mock.patch.object(TTSClient, "from_env") as factory:
        service.handle_command(command, 10001, "http://onebot", "")
    factory.assert_not_called()
    assert "用法" in send.call_args.args[2]["message"]
    assert service.enqueue_if_enabled(10001, "正文", [], None, "http://onebot", "") is False


def test_missing_configuration_keeps_mode_off() -> None:
    """验证服务或 Gemini 凭据缺失时不启用模式。

    Returns:
        None: 无返回值。

    Raises:
        None: 本函数不主动抛出异常；测试结果由断言验证。
    """
    service, send, _, _, _ = _service()
    for env in ({}, {"TTS_API_BASE": "http://tts"}):
        with mock.patch.dict(mode_module.os.environ, env, clear=True):
            service.handle_command("/ttsmode on", 10001, "http://onebot", "")
        assert "设置失败" in send.call_args.args[2]["message"]
        assert service.enqueue_if_enabled(10001, "正文", [], None, "http://onebot", "") is False


def test_snapshot_preserves_original_group_images_and_configuration() -> None:
    """验证排队后切换模式或修改图片列表不会改变已提交回复。

    Returns:
        None: 无返回值。

    Raises:
        None: 本函数不主动抛出异常；测试结果由断言验证。
    """
    service, send, executor, tts, rewriter = _service()
    _enable(service, tts, rewriter)
    images = [("aW1hZ2U=", "image/png")]
    service.enqueue_if_enabled(10001, "原回复", images, None, "http://original", "old-token")
    task, job = executor.submit.call_args.args
    images.clear()
    service.handle_command("/ttsmode off", 10001, "http://new", "new-token")
    send.reset_mock()
    task(job)
    assert send.call_args_list[0].args[0] == "http://original"
    assert send.call_args_list[0].args[3] == "old-token"
    assert send.call_args_list[0].args[2]["group_id"] == 10001
    message = send.call_args_list[0].args[2]["message"]
    assert message.startswith(rewriter.rewrite.return_value)
    assert "[CQ:image,file=base64://aW1hZ2U=" in message
    record = send.call_args_list[1].args[2]["message"][0]
    assert record["type"] == "record"
    assert base64.b64decode(record["data"]["file"].removeprefix("base64://")) == WAV_BYTES
    tts.synthesize.assert_called_once_with(rewriter.rewrite.return_value)
    rewriter.rewrite.assert_called_once_with("原回复")


@pytest.mark.parametrize(("original", "cleaned", "urls"), [
    ('正文 ![图](https://img.test/a.png "标题")', "正文", ("https://img.test/a.png",)),
    ("正文 https://img.test/a.jpg?token=1", "正文", ("https://img.test/a.jpg?token=1",)),
    ("![图](https://img.test/a_(1).png)", "", ("https://img.test/a_(1).png",)),
    ("https://img.test/a.png。", "。", ("https://img.test/a.png",)),
    ("网页 https://example.com/page", "网页 https://example.com/page", ()),
    ("![图](https://img.test/a.png)\nhttps://img.test/a.png", "", ("https://img.test/a.png",)),
])
def test_image_addresses_are_removed_without_treating_webpages_as_images(
    original: str, cleaned: str, urls: tuple[str, ...]
) -> None:
    """验证图片地址移出正文，普通网页链接保留。

    Args:
        original (str): 含链接的原回复。
        cleaned (str): 预期正文。
        urls (tuple[str, ...]): 预期图片下载列表。

    Returns:
        None: 无返回值。

    Raises:
        None: 本函数不主动抛出异常；测试结果由断言验证。
    """
    assert TTSModeService._split_images(original) == (cleaned, urls)


def test_markdown_image_is_sent_and_never_reaches_models() -> None:
    """验证 Markdown 图片进入原图片发送器，地址不进入 Flash 或 TTS。

    Returns:
        None: 无返回值。

    Raises:
        None: 本函数不主动抛出异常；测试结果由断言验证。
    """
    service, send, _, tts, rewriter = _service()
    storage = mock.Mock(spec=ImageStorageManager)
    storage.is_generated_path.return_value = False
    storage.save_remote_image.return_value = StoredImage(Path("image.png"), "image/png", "aW1hZ2U=")
    service._deliver(TTSReplyJob(
        10001, "给您看这张图。 ![图片](https://img.test/a.png)", (), storage,
        tts, rewriter, "http://onebot", "token",
    ))
    rewriter.rewrite.assert_called_once_with("给您看这张图。")
    tts.synthesize.assert_called_once_with(rewriter.rewrite.return_value)
    storage.save_remote_image.assert_called_once_with("https://img.test/a.png")
    assert "[CQ:image" in send.call_args_list[0].args[2]["message"]


@pytest.mark.parametrize("text", ["", "（图片已发送）", "。"])
def test_image_only_reply_does_not_create_speech(text: str) -> None:
    """验证纯图片回复不会凭空生成一段旁白或额外模型请求。

    Args:
        text (str): 空正文或原流程的图片发送占位文字。

    Returns:
        None: 无返回值。

    Raises:
        None: 本函数不主动抛出异常；测试结果由断言验证。
    """
    service, send, _, tts, rewriter = _service()
    service._deliver(TTSReplyJob(
        10001, text, (("aW1hZ2U=", "image/png"),), None,
        tts, rewriter, "http://onebot", "token",
    ))
    tts.synthesize.assert_not_called()
    rewriter.rewrite.assert_not_called()
    send.assert_called_once()
    assert send.call_args.args[2]["message"].startswith("[CQ:image")


@pytest.mark.parametrize("error", [
    errors.ClientError(429, {"message": "quota", "status": "RESOURCE_EXHAUSTED"}),
    AssertionError("无日语结果"), httpx.ReadTimeout("超时"),
    GoogleAuthError("Vertex AI 凭据不可用"),
])
def test_rewrite_failure_reports_error_and_keeps_images(error: Exception) -> None:
    """验证改写失败明确提示并保留图片，不发送原文或语音。

    Args:
        error (Exception): 模拟的改写失败。

    Returns:
        None: 无返回值。

    Raises:
        None: 本函数不主动抛出异常；测试结果由断言验证。
    """
    service, send, _, tts, rewriter = _service()
    rewriter.rewrite.side_effect = error
    service._deliver(TTSReplyJob(
        10001, "原回复不能替代失败结果", (("aW1hZ2U=", "image/png"),), None,
        tts, rewriter, "http://onebot", "token",
    ))
    send.assert_called_once()
    message = send.call_args.args[2]["message"]
    assert "TTS 模式日语改写失败" in message
    assert "[CQ:image" in message
    assert "原回复不能替代失败结果" not in message
    tts.synthesize.assert_not_called()


def test_synthesis_failure_reports_error_before_any_japanese_delivery() -> None:
    """验证合成失败时没有先发日语后缺失语音的半成品。

    Returns:
        None: 无返回值。

    Raises:
        None: 本函数不主动抛出异常；测试结果由断言验证。
    """
    service, send, _, tts, rewriter = _service()
    rewriter.rewrite.return_value = "日" * 2001
    http_post = mock.Mock()
    actual_tts = TTSClient("http://tts", http_post=http_post)
    service._deliver(TTSReplyJob(
        10001, "原回复", (), None, actual_tts, rewriter, "http://onebot", "token",
    ))
    send.assert_called_once()
    assert "TTS 模式语音生成失败" in send.call_args.args[2]["message"]
    assert rewriter.rewrite.return_value not in send.call_args.args[2]["message"]
    http_post.assert_not_called()


def test_image_download_failure_is_reported_and_preserves_prepared_images() -> None:
    """验证图片下载器的真实异常类型会转为明确提示，并保留已有图片。

    Returns:
        None: 无返回值。

    Raises:
        None: 本函数不主动抛出异常；测试结果由断言验证。
    """
    service, send, _, tts, rewriter = _service()
    storage = mock.Mock(spec=ImageStorageManager)
    storage.is_generated_path.return_value = False
    storage.save_remote_image.side_effect = RuntimeError("HTTP 404")
    service._deliver(TTSReplyJob(
        10001, "正文 ![图](https://img.test/failed.png)",
        (("aW1hZ2U=", "image/png"),), storage,
        tts, rewriter, "http://onebot", "token",
    ))
    send.assert_called_once()
    message = send.call_args.args[2]["message"]
    assert "TTS 模式图片处理失败" in message
    assert "[CQ:image" in message
    rewriter.rewrite.assert_not_called()
    tts.synthesize.assert_not_called()


def test_group_toggle_command_never_enters_agent() -> None:
    """验证真实群回调能开启再关闭本群模式，并立即结束命令回合。

    Returns:
        None: 无返回值。

    Raises:
        None: 本函数不主动抛出异常；测试结果由断言验证。
    """
    _, _, executor, tts, rewriter = _service()
    handler = _handler(text="/ttsmode")
    with (
        mock.patch.object(TTSClient, "from_env", return_value=tts),
        mock.patch.object(JapaneseReplyRewriter, "from_env", return_value=rewriter),
        mock.patch.object(mode_module, "_TTS_MODE_EXECUTOR", executor),
        mock.patch.object(qq_group_bot, "_call_onebot_action") as send,
    ):
        handler.do_POST()
        assert "已开启" in send.call_args.args[2]["message"]
        handler.do_POST()
        assert "已关闭" in send.call_args.args[2]["message"]
    handler.agent.chat_once_stream.assert_not_called()
    handler.agent.consume_generated_images.assert_not_called()
    executor.submit.assert_not_called()
    assert handler._send_no_content.call_count == 2


def test_mode_onebot_business_failure_is_visible(capsys: pytest.CaptureFixture[str]) -> None:
    """验证语音投递遇到 NapCat 业务失败时不会被当成发送成功。

    Args:
        capsys (pytest.CaptureFixture[str]): 捕获后台异常输出。

    Returns:
        None: 无返回值。

    Raises:
        None: 本函数不主动抛出异常；测试结果由断言验证。
    """
    _, _, _, tts, rewriter = _service()
    response = mock.MagicMock(status=200)
    response.read.return_value = b'{"status":"failed","retcode":1200,"message":"denied"}'
    response.__enter__.return_value = response
    with (
        ThreadPoolExecutor(max_workers=1) as executor,
        mock.patch.object(qq_group_bot, "urlopen", return_value=response),
    ):
        service = TTSModeService(
            qq_group_bot._call_onebot_action, QQBotHandler._compose_group_message, executor
        )
        service._groups[10001] = (tts, rewriter)
        assert service.enqueue_if_enabled(10001, "正文", [], None, "http://onebot", "") is True
        executor.shutdown(wait=True)
    output = capsys.readouterr().err
    assert "[TTSMode] 后台回复投递失败" in output
    assert "retcode=1200" in output


def test_off_mode_preserves_original_reply_and_agent_thread() -> None:
    """验证默认关闭与已创建但关闭的服务均保留原回复出口。

    Returns:
        None: 无返回值。

    Raises:
        None: 本函数不主动抛出异常；测试结果由断言验证。
    """
    for service in (None, _service()[0]):
        handler = _handler("原回复 https://img.test/a.png")
        with (
            mock.patch.object(QQBotHandler, "tts_mode", service),
            mock.patch.object(qq_group_bot, "_send_group_msg") as send,
        ):
            handler.do_POST()
        send.assert_called_once_with("http://onebot", 10001, "原回复 https://img.test/a.png", "token")
        assert handler.agent.chat_once_stream.call_count == 1
        assert handler.agent.chat_once_stream.call_args.kwargs == {"thread_id": "original-thread"}
        handler._send_no_content.assert_called_once()


def test_existing_image_tags_and_generated_images_keep_their_original_path(tmp_path: Path) -> None:
    """验证旧 IMAGE 标签与生成图片仍由现有代码准备，再交给 TTS 出口。

    Args:
        tmp_path (Path): 生成图片临时目录。

    Returns:
        None: 无返回值。

    Raises:
        None: 本函数不主动抛出异常；测试结果由断言验证。
    """
    service, _, executor, tts, rewriter = _service()
    _enable(service, tts, rewriter)
    generated = tmp_path / "generated.png"
    generated.write_bytes(b"generated-image")
    handler = _handler("请看图片。\n[IMAGE]https://img.test/a.png[/IMAGE]")
    handler.agent.consume_generated_images.return_value = [GeneratedImage(generated, "image/png", "图")]
    storage = mock.Mock(spec=ImageStorageManager)
    storage.is_generated_path.return_value = False
    storage.save_remote_image.return_value = StoredImage(Path("remote.png"), "image/png", "cmVtb3Rl")
    handler.image_storage = storage
    with (
        mock.patch.object(QQBotHandler, "tts_mode", service),
        mock.patch.object(QQBotHandler, "image_storage", storage),
        mock.patch.object(qq_group_bot, "_send_group_msg") as normal_send,
    ):
        handler.do_POST()
    normal_send.assert_not_called()
    storage.save_remote_image.assert_called_once_with("https://img.test/a.png")
    job = executor.submit.call_args.args[1]
    assert job.original == "请看图片。"
    assert job.images == ((base64.b64encode(b"generated-image").decode(), "image/png"), ("cmVtb3Rl", "image/png"))
    assert handler.agent.chat_once_stream.call_count == 1
    handler._send_no_content.assert_called_once()


def test_agent_error_is_not_rewritten_even_when_mode_is_on() -> None:
    """验证主 Agent 的配置错误继续使用已有提示，不触发语音流程。

    Returns:
        None: 无返回值。

    Raises:
        None: 本函数不主动抛出异常；测试结果由断言验证。
    """
    service, _, executor, tts, rewriter = _service()
    _enable(service, tts, rewriter)
    handler = _handler()
    handler.agent.chat_once_stream.side_effect = AssertionError("原配置错误")
    with (
        mock.patch.object(QQBotHandler, "tts_mode", service),
        mock.patch.object(qq_group_bot, "_send_group_msg") as send,
    ):
        handler.do_POST()
    executor.submit.assert_not_called()
    assert send.call_args.args[2] == "（配置错误）原配置错误"


def test_mode_command_respects_existing_command_whitelist() -> None:
    """验证未授权用户不能创建模式服务或更改开关。

    Returns:
        None: 无返回值。

    Raises:
        None: 本函数不主动抛出异常；测试结果由断言验证。
    """
    handler = _handler()
    handler.bot_cfg.cmd_allowed_users = (99999,)
    with mock.patch.object(qq_group_bot, "_send_group_msg") as send:
        assert handler._handle_commands(10001, 20002, "/ttsmode on") is True
    assert QQBotHandler.tts_mode is None
    assert send.call_args.args[2] == "无权执行命令（需在白名单内）。"


def test_slow_mode_request_releases_post_lock_and_other_group_runs_normally() -> None:
    """验证慢速改写不占用 POST 锁，其他群仍能走正常回复流程。

    Returns:
        None: 无返回值。

    Raises:
        None: 本函数不主动抛出异常；测试结果由断言验证。
    """
    started = Event()
    release = Event()
    service, send, _, tts, rewriter = _service()

    def slow_rewrite(original: str) -> str:
        """等待主线程放行，模拟较慢的 Flash 请求。

        Args:
            original (str): 本轮原回复。

        Returns:
            str: 模拟的日语正文。

        Raises:
            AssertionError: 当请求未释放锁导致测试超时时抛出。
        """
        started.set()
        assert release.wait(5), "TTS 模式占用了群消息处理锁"
        return "そなた、参りましょうー。"

    rewriter.rewrite.side_effect = slow_rewrite
    with (
        ThreadPoolExecutor(max_workers=1) as executor,
        mock.patch.object(service, "_executor", executor),
        mock.patch.object(QQBotHandler, "tts_mode", service),
        mock.patch.object(qq_group_bot, "_send_group_msg") as normal_send,
    ):
        _enable(service, tts, rewriter)
        send.reset_mock()
        try:
            handler = _handler()
            handler.do_POST()
            assert started.wait(2)
            handler._send_no_content.assert_called_once()
            assert QQBotHandler._post_lock.acquire(blocking=False)
            QQBotHandler._post_lock.release()
            other = _handler("其他群原回复", 10002)
            other.do_POST()
            normal_send.assert_called_once_with("http://onebot", 10002, "其他群原回复", "token")
            assert send.call_count == 0
            assert handler.agent.chat_once_stream.call_count == 1
        finally:
            release.set()
        executor.shutdown(wait=True)
    assert send.call_count == 2
