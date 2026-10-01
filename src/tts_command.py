"""独立处理 TTS 命令、WAV 生成与 NapCat 语音消息发送。"""

from __future__ import annotations

import base64
import os
import shlex
import sys
import traceback
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from typing import Callable
from urllib.parse import urljoin, urlsplit

import httpx


TTS_API_BASE_ENV = "TTS_API_BASE"
_TTS_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-command")
HttpPost = Callable[..., httpx.Response]
RecordMessage = list[dict[str, dict[str, str]]]
OneBotAction = Callable[[str, str, dict[str, object], str], dict[str, object]]


class TTSClient:
    """调用配置的 TTS 服务，使用服务端默认模型和角色生成 WAV。"""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 300.0,
        http_post: HttpPost | None = None,
    ) -> None:
        """初始化 TTS 客户端。

        Args:
            base_url (str): TTS 服务的 HTTP 或 HTTPS 基地址。
            timeout_seconds (float): 生成请求超时秒数，默认 300 秒。
            http_post (HttpPost | None): 可注入的 HTTP POST 函数。

        Returns:
            None: 构造函数无返回值。

        Raises:
            AssertionError: 当服务地址或超时时间非法时抛出。
        """
        normalized_url = base_url.strip().rstrip("/")
        try:
            parsed_url = urlsplit(normalized_url)
        except ValueError as error:
            raise AssertionError("TTS_API_BASE 服务地址非法") from error
        assert parsed_url.scheme in {"http", "https"} and parsed_url.netloc, (
            "TTS_API_BASE 必须是完整的 HTTP 或 HTTPS 服务地址"
        )
        assert not parsed_url.query and not parsed_url.fragment, (
            "TTS_API_BASE 不可包含查询参数或片段"
        )
        assert timeout_seconds > 0, "timeout_seconds 必须大于 0"
        self._base_url = normalized_url
        self._timeout_seconds = timeout_seconds
        self._http_post = http_post if http_post is not None else httpx.post

    @classmethod
    def from_env(cls) -> TTSClient | None:
        """从环境变量创建客户端，未配置时禁用 TTS。

        Returns:
            TTSClient | None: 已配置的客户端，未配置或留空时为 None。

        Raises:
            AssertionError: 当已配置的服务地址非法时抛出。
        """
        base_url = os.environ.get(TTS_API_BASE_ENV, "").strip()
        if not base_url:
            return None
        return cls(base_url)

    def synthesize(self, text: str) -> bytes:
        """将文本交给 TTS 服务并取得 WAV 字节。

        Args:
            text (str): 要合成的原文，长度为 1 至 2000 个字符。

        Returns:
            bytes: 服务返回的 WAV 音频。

        Raises:
            AssertionError: 当文本、响应状态或 WAV 格式不符合约定时抛出。
            httpx.HTTPError: 当 TTS 请求失败或超时时抛出。
        """
        assert text.strip(), "TTS 文本不能为空"
        assert len(text) <= 2000, "TTS 文本不能超过 2000 个字符"
        response = self._http_post(
            urljoin(self._base_url + "/", "v1/audio/speech"),
            json={"input": text, "response_format": "wav"},
            timeout=httpx.Timeout(self._timeout_seconds, connect=10.0),
        )
        response.raise_for_status()
        assert response.status_code == 200, "TTS 服务必须返回 HTTP 200"
        content_type = response.headers.get("content-type", "").split(";", 1)[0]
        assert content_type.strip().lower() == "audio/wav", (
            "TTS 服务必须返回 audio/wav"
        )
        audio = response.content
        assert len(audio) > 12 and audio[:4] == b"RIFF" and audio[8:12] == b"WAVE", (
            "TTS 服务未返回有效的 WAV 音频"
        )
        return audio


class TTSCommandHandler:
    """在独立工作线程中生成并发送语音，复用现有 OneBot action 调用。"""

    def __init__(
        self,
        call_onebot_action: OneBotAction,
        executor: Executor | None = None,
    ) -> None:
        """初始化语音命令处理器。

        Args:
            call_onebot_action (OneBotAction): 校验业务响应的 OneBot 调用函数。
            executor (Executor | None): 可注入的任务执行器，默认独立单线程。

        Returns:
            None: 构造函数无返回值。

        Raises:
            None: 本函数不主动抛出异常。
        """
        self._call_onebot_action = call_onebot_action
        self._executor = executor if executor is not None else _TTS_EXECUTOR

    def handle(
        self,
        command_text: str,
        group_id: int,
        onebot_api_base: str,
        access_token: str = "",
    ) -> None:
        """校验命令并提交语音任务，未配置服务时静默结束。

        Args:
            command_text (str): 完整的 /tts "文本" 命令。
            group_id (int): 接收语音的原群号。
            onebot_api_base (str): NapCat HTTP API 基地址。
            access_token (str): NapCat API Token，可为空。

        Returns:
            None: 校验完成后立即返回，不等待语音生成。

        Raises:
            AssertionError: 当群号不是正整数时抛出。
            RuntimeError: 当任务执行器已关闭或 OneBot 错误提示发送失败时抛出。
            OSError: 当 OneBot 错误提示请求失败时抛出。
        """
        assert group_id > 0, "group_id 必须为正整数"
        try:
            client = TTSClient.from_env()
            if client is None:
                return
            text = self._parse_text(command_text)
        except AssertionError as error:
            self._call_onebot_action(
                onebot_api_base,
                "send_group_msg",
                {"group_id": group_id, "message": f"TTS 参数错误：{error}"},
                access_token,
            )
            return
        future = self._executor.submit(
            self._generate_and_send,
            client,
            text,
            group_id,
            onebot_api_base,
            access_token,
        )
        future.add_done_callback(self._report_failure)

    @staticmethod
    def _parse_text(command_text: str) -> str:
        """提取双引号包裹的单个文本参数，保留空格、换行与转义引号。

        Args:
            command_text (str): 完整命令文本。

        Returns:
            str: 用于合成的文本原文。

        Raises:
            AssertionError: 当命令格式非法、文本为空或超过长度限制时抛出。
        """
        usage = '用法：/tts "文本"（最多 2000 个字符）'
        argument = command_text[len("/tts") :].strip()
        assert argument.startswith('"') and argument.endswith('"'), usage
        try:
            parts = shlex.split(command_text)
        except ValueError as error:
            raise AssertionError(usage) from error
        assert len(parts) == 2 and parts[0] == "/tts", usage
        text = parts[1]
        assert text.strip(), "TTS 文本不能为空"
        assert len(text) <= 2000, "TTS 文本不能超过 2000 个字符"
        return text

    def _generate_and_send(
        self,
        client: TTSClient,
        text: str,
        group_id: int,
        onebot_api_base: str,
        access_token: str,
    ) -> None:
        """生成 WAV 并以 Base64 语音消息发送到原群。

        Args:
            client (TTSClient): 本次任务的 TTS 客户端。
            text (str): 已校验的合成文本。
            group_id (int): 接收语音的群号。
            onebot_api_base (str): NapCat HTTP API 基地址。
            access_token (str): NapCat API Token。

        Returns:
            None: 发送完成后无返回值。

        Raises:
            RuntimeError: 当 NapCat 业务响应失败时抛出，由任务回调记录。
            OSError: 当 NapCat 请求失败时抛出，由任务回调记录。
        """
        try:
            audio = client.synthesize(text)
        except (AssertionError, httpx.HTTPError) as error:
            message: str | RecordMessage = f"TTS 生成失败：{error}"
        else:
            encoded_audio = base64.b64encode(audio).decode("ascii")
            message = [
                {"type": "record", "data": {"file": f"base64://{encoded_audio}"}}
            ]
        self._call_onebot_action(
            onebot_api_base,
            "send_group_msg",
            {"group_id": group_id, "message": message},
            access_token,
        )

    @staticmethod
    def _report_failure(future: Future[None]) -> None:
        """将后台任务未处理的异常完整输出到错误日志。

        Args:
            future (Future[None]): 已完成的语音任务。

        Returns:
            None: 记录失败信息后无返回值。

        Raises:
            None: 本函数不主动抛出异常。
        """
        error = future.exception()
        if error is not None:
            sys.stderr.write("[TTS] 语音命令执行失败\n")
            traceback.print_exception(error, file=sys.stderr)
