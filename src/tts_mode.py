"""独立管理群 TTS 模式、日语改写和后台回复投递。"""

from __future__ import annotations

import base64
import os
import re
import sys
import traceback
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from threading import Lock
from typing import Callable, Sequence
from urllib.parse import urlsplit

import httpx
from google import genai
from google.auth.exceptions import GoogleAuthError
from google.genai import errors, types

from image_storage import ImageStorageManager
from src.tts_command import OneBotAction, TTSClient


TTS_REWRITE_MODEL_ENV = "TTS_REWRITE_MODEL"
TTS_REWRITE_PROMPT_ENV = "TTS_REWRITE_PROMPT_FILE"
DEFAULT_REWRITE_MODEL = "gemini-3.8-flash"
DEFAULT_REWRITE_PROMPT = (
    Path(__file__).resolve().parent.parent / "prompts" / "tts" / "tts_yoshino_prompt.txt"
)
_TTS_MODE_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-mode")
_MARKDOWN_IMAGE = re.compile(
    r'!\[[^\]]*\]\((https?://(?:[^\s()]|\([^\s()]*\))+)(?:\s+"[^"]*")?\)',
    re.IGNORECASE,
)
_HTTP_URL = re.compile(r"https?://(?:[^\s<>\[\]()]|\([^\s()]*\))+", re.IGNORECASE)
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".avif", ".bmp")
MessageComposer = Callable[[str, Sequence[tuple[str, str]]], str]
GeminiClientFactory = Callable[[], genai.Client]


class JapaneseReplyRewriter:
    """通过独立、无历史的 Gemini 请求压缩改写日语回复。

    Args:
        model (str): Google 原生 API 模型名称。
        system_prompt (str): 独立的改写提示词。
        client_factory (GeminiClientFactory): 客户端创建函数。

    Returns:
        None: 构造函数无返回值。

    Raises:
        AssertionError: 当模型或提示词为空时抛出。
    """

    def __init__(
        self, model: str, system_prompt: str, client_factory: GeminiClientFactory
    ) -> None:
        """保存本次模式启用时的模型、提示词和连接配置。

        Args:
            model (str): Google 原生 API 模型名称。
            system_prompt (str): 独立的日语改写提示词。
            client_factory (GeminiClientFactory): 创建客户端的函数。

        Returns:
            None: 初始化无返回值。

        Raises:
            AssertionError: 当模型或提示词为空时抛出。
        """
        assert model.strip(), "TTS_REWRITE_MODEL 不能为空"
        assert system_prompt.strip(), "TTS 日语改写提示词不能为空"
        self._model = model.strip()
        self._system_prompt = system_prompt.strip()
        self._client_factory = client_factory

    @classmethod
    def from_env(cls) -> JapaneseReplyRewriter:
        """读取改写配置，复用现有 Gemini Key 或 Vertex AI 凭据。

        Returns:
            JapaneseReplyRewriter: 配置已校验的独立改写器。

        Raises:
            AssertionError: 当 Gemini 凭据、模型或提示词非法时抛出。
            OSError: 当提示词文件无法读取时抛出。
        """
        api_key = (
            os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY")
            or ""
        ).strip()
        http_options = types.HttpOptions(timeout=120000)
        if api_key:
            factory = partial(
                genai.Client,
                vertexai=False,
                api_key=api_key,
                http_options=http_options,
            )
        else:
            use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower()
            project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
            location = os.environ.get("GOOGLE_CLOUD_LOCATION", "").strip()
            assert use_vertex in {"1", "true", "yes", "on"} and project and location, (
                "请配置 Gemini API Key，或 Vertex AI 的项目和区域"
            )
            factory = partial(
                genai.Client,
                vertexai=True,
                project=project,
                location=location,
                http_options=http_options,
            )
        prompt_path = Path(
            os.environ.get(TTS_REWRITE_PROMPT_ENV, "").strip()
            or DEFAULT_REWRITE_PROMPT
        ).expanduser()
        return cls(
            os.environ.get(TTS_REWRITE_MODEL_ENV, DEFAULT_REWRITE_MODEL),
            prompt_path.read_text(encoding="utf-8"),
            factory,
        )

    def rewrite(self, original: str) -> str:
        """仅改写本轮正文，不调用主 Agent 或追加任何对话记录。

        Args:
            original (str): 已移除图片地址的原回复正文。

        Returns:
            str: 同时用于群文本和语音合成的日语正文。

        Raises:
            AssertionError: 当输入为空、生成未完成或输出为空时抛出。
            errors.APIError: 当 Gemini 请求失败时抛出。
            GoogleAuthError: 当 Vertex AI 凭据不可用时抛出。
            httpx.HTTPError: 当 Gemini 连接失败或超时时抛出。
        """
        assert original.strip(), "待改写回复不能为空"
        client = self._client_factory()
        try:
            response = client.models.generate_content(
                model=self._model,
                contents="待改写回答：\n" + original,
                config=types.GenerateContentConfig(
                    system_instruction=self._system_prompt,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.LOW
                    ),
                    max_output_tokens=2048,
                ),
            )
        finally:
            client.close()
        assert response.candidates, "日语改写未返回结果"
        assert response.candidates[0].finish_reason == types.FinishReason.STOP, (
            "日语改写未正常完成"
        )
        text = (response.text or "").strip()
        assert text, "日语改写结果为空"
        return text


@dataclass(frozen=True)
class TTSReplyJob:
    """保存回复正文、图片、目标群及启用时的配置快照。

    Args:
        group_id (int): 接收消息的群号。
        original (str): 主 Agent 已生成的正文。
        images (tuple[tuple[str, str], ...]): 原流程已准备的图片。
        storage (ImageStorageManager | None): 原流程的图片存储器。
        tts (TTSClient): 服务配置快照。
        rewriter (JapaneseReplyRewriter): 改写配置快照。
        api_base (str): NapCat API 地址。
        access_token (str): NapCat 凭据，不显示在对象表示中。

    Returns:
        None: 无额外返回值。

    Raises:
        None: 本对象或回调不主动抛出异常。
    """

    group_id: int
    original: str
    images: tuple[tuple[str, str], ...]
    storage: ImageStorageManager | None
    tts: TTSClient
    rewriter: JapaneseReplyRewriter
    api_base: str
    access_token: str = field(repr=False)


class TTSModeService:
    """按群开关模式，在独立串行队列中发送日语、图片及语音。

    Args:
        call_onebot_action (OneBotAction): 现有 OneBot 业务校验函数。
        compose_message (MessageComposer): 现有消息组合函数。
        executor (Executor | None): 可注入的后台执行器。

    Returns:
        None: 构造函数无返回值。

    Raises:
        None: 初始化不访问外部服务或配置文件。
    """

    def __init__(
        self,
        call_onebot_action: OneBotAction,
        compose_message: MessageComposer,
        executor: Executor | None = None,
    ) -> None:
        """初始化投递服务；默认没有开启任何群。

        Args:
            call_onebot_action (OneBotAction): 现有 OneBot 业务校验函数。
            compose_message (MessageComposer): 现有文本与图片组合函数。
            executor (Executor | None): 可注入的后台执行器。

        Returns:
            None: 初始化无返回值。

        Raises:
            None: 本函数不主动抛出异常。
        """
        self._call_onebot_action = call_onebot_action
        self._compose_message = compose_message
        self._executor = executor if executor is not None else _TTS_MODE_EXECUTOR
        self._groups: dict[int, tuple[TTSClient, JapaneseReplyRewriter]] = {}
        self._lock = Lock()

    def handle_command(
        self, command: str, group_id: int, api_base: str, access_token: str
    ) -> None:
        """切换当前群的模式，支持无参数切换和显式 on/off。

        Args:
            command (str): 完整的 /ttsmode 命令。
            group_id (int): 当前群号。
            api_base (str): NapCat API 地址。
            access_token (str): NapCat 凭据。

        Returns:
            None: 发送命令处理结果后返回。

        Raises:
            AssertionError: 当群号非法时抛出。
            OSError: 当结果消息发送失败时抛出。
            RuntimeError: 当 NapCat 业务响应失败时抛出。
        """
        assert group_id > 0, "group_id 必须为正整数"
        try:
            parts = command.split()
            assert len(parts) in {1, 2} and parts[0] == "/ttsmode", (
                "用法：/ttsmode 或 /ttsmode on|off"
            )
            with self._lock:
                enabled = group_id in self._groups
            if len(parts) == 1:
                enable = not enabled
            else:
                argument = parts[1].lower()
                assert argument in {"on", "off", "开启", "关闭"}, (
                    "用法：/ttsmode 或 /ttsmode on|off"
                )
                enable = argument in {"on", "开启"}
            if enable and not enabled:
                tts = TTSClient.from_env()
                assert tts is not None, "请先配置 TTS_API_BASE"
                rewriter = JapaneseReplyRewriter.from_env()
                with self._lock:
                    self._groups[group_id] = (tts, rewriter)
            elif not enable:
                with self._lock:
                    self._groups.pop(group_id, None)
            message = (
                "本群 TTS 模式已开启：回复将改写为日语文本＋语音，图片照常发送。"
                if enable else "本群 TTS 模式已关闭。"
            )
        except (AssertionError, OSError, ValueError) as error:
            message = f"TTS 模式设置失败：{error}"
        self._send(group_id, message, api_base, access_token)

    def enqueue_if_enabled(
        self,
        group_id: int,
        original: str,
        images: Sequence[tuple[str, str]],
        storage: ImageStorageManager | None,
        api_base: str,
        access_token: str,
    ) -> bool:
        """为已开启的群排队投递，关闭时立即返回原流程。

        Args:
            group_id (int): 接收消息的群号。
            original (str): 主 Agent 的最终正文。
            images (Sequence[tuple[str, str]]): 已准备的图片 Base64 与 MIME。
            storage (ImageStorageManager | None): 原流程图片存储器。
            api_base (str): NapCat API 地址。
            access_token (str): NapCat 凭据。

        Returns:
            bool: 已接管发送时为 True，当前群关闭时为 False。

        Raises:
            RuntimeError: 当后台执行器已关闭时抛出。
        """
        with self._lock:
            config = self._groups.get(group_id)
        if config is None:
            return False
        job = TTSReplyJob(
            group_id, original, tuple(images), storage,
            config[0], config[1], api_base, access_token,
        )
        future = self._executor.submit(self._deliver, job)
        future.add_done_callback(self._report_failure)
        return True

    @staticmethod
    def _split_images(text: str) -> tuple[str, tuple[str, ...]]:
        """提取 Markdown 图片和带图片扩展名的 HTTP 链接。

        Args:
            text (str): 原流程清理 IMAGE 标签后的正文。

        Returns:
            tuple[str, tuple[str, ...]]: 无图片地址的正文与去重后的图片地址。

        Raises:
            ValueError: 当链接无法解析时抛出。
        """
        urls: list[str] = []

        def take_markdown(match: re.Match[str]) -> str:
            """取出 Markdown 图片地址并移除整段标记。

            Args:
                match (re.Match[str]): Markdown 图片匹配。

            Returns:
                str: 用空串替换图片标记。

            Raises:
                None: 本回调不主动抛出异常。
            """
            urls.append(match.group(1))
            return ""

        def take_url(match: re.Match[str]) -> str:
            """仅移除可由扩展名识别的直接图片链接。

            Args:
                match (re.Match[str]): HTTP 链接匹配。

            Returns:
                str: 普通链接原文，或图片链接后的标点。

            Raises:
                ValueError: 当链接无法解析时抛出。
            """
            raw = match.group()
            url = raw.rstrip("。，；、！？;,.!?")
            if urlsplit(url).path.lower().endswith(_IMAGE_EXTENSIONS):
                urls.append(url)
                return raw[len(url):]
            return raw

        cleaned = _MARKDOWN_IMAGE.sub(take_markdown, text)
        cleaned = _HTTP_URL.sub(take_url, cleaned).strip()
        return cleaned, tuple(dict.fromkeys(urls))

    def _deliver(self, job: TTSReplyJob) -> None:
        """先完成改写和合成，再发送同一份日语正文、图片和语音。

        Args:
            job (TTSReplyJob): 提交时捕获的回复和目标配置。

        Returns:
            None: 投递结束后返回，纯图片回复不调用模型或 TTS。

        Raises:
            OSError: 当 NapCat 发送失败时抛出。
            RuntimeError: 当 NapCat 业务响应失败时抛出。
        """
        images = list(job.images)
        stage = "图片处理"
        try:
            original, urls = self._split_images(job.original)
            for url in urls:
                assert job.storage is not None, "图片存储管理器尚未配置"
                if job.storage.is_generated_path(url):
                    continue
                saved = job.storage.save_remote_image(url)
                assert saved is not None, "未能下载回复图片"
                images.append((saved.base64_data, saved.mime_type))
            if original == "（图片已发送）" and images:
                original = ""
            if images and not original.strip(" \t\r\n。，；、！？;,.!?…ー"):
                original = ""
            if not original:
                assert images, "回复正文和图片均为空"
                message = self._compose_message("", images)
                audio = None
            else:
                stage = "日语改写"
                japanese = job.rewriter.rewrite(original)
                stage = "语音生成"
                audio = job.tts.synthesize(japanese)
                message = self._compose_message(japanese, images)
        except errors.APIError as error:
            message = self._compose_message(
                f"TTS 模式{stage}失败：Gemini HTTP {error.code}。", images
            )
            audio = None
        except (
            AssertionError, httpx.HTTPError, OSError, ValueError, RuntimeError,
            GoogleAuthError,
        ) as error:
            message = self._compose_message(f"TTS 模式{stage}失败：{error}", images)
            audio = None
        self._send(job.group_id, message, job.api_base, job.access_token)
        if audio is not None:
            encoded = base64.b64encode(audio).decode("ascii")
            record = [{"type": "record", "data": {"file": f"base64://{encoded}"}}]
            self._send(job.group_id, record, job.api_base, job.access_token)

    def _send(
        self, group_id: int, message: object, api_base: str, access_token: str
    ) -> None:
        """通过现有 OneBot 调用发送消息，不保存投递记录。

        Args:
            group_id (int): 目标群号。
            message (object): 文本或 OneBot 消息段。
            api_base (str): NapCat API 地址。
            access_token (str): NapCat 凭据。

        Returns:
            None: 无返回值。

        Raises:
            OSError: 当请求失败时抛出。
            RuntimeError: 当业务响应失败时抛出。
        """
        self._call_onebot_action(
            api_base, "send_group_msg",
            {"group_id": group_id, "message": message}, access_token,
        )

    @staticmethod
    def _report_failure(future: Future[None]) -> None:
        """将后台投递的未处理异常输出到标准错误。

        Args:
            future (Future[None]): 已完成的投递任务。

        Returns:
            None: 本函数不记录正常投递内容。

        Raises:
            None: 本函数不主动抛出异常。
        """
        error = future.exception()
        if error is not None:
            sys.stderr.write("[TTSMode] 后台回复投递失败\n")
            traceback.print_exception(error, file=sys.stderr)
