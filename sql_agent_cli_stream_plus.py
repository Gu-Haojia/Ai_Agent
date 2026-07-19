"""
增强版：流式 Agent（多轮工具、先 Tool 后 Agent、强化综合总结）。

特性：
- 多轮工具：允许在工具结果后再次调用工具（由模型自行决定），直到不再需要工具为止；
- 输出顺序：先输出 Tool 调用行，再开始 Agent 流式内容；
- 强化系统提示：禁止生搬硬套搜索结果，要求围绕用户意图综合、提炼、给出可执行建议；
- REPL 仅在输入 :help 时显示帮助；
- 历史按时间顺序输出（timetravel 风格），支持索引回放。

"""

from __future__ import annotations
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Callable, Iterable, Match, Optional, Sequence, Union, Any
from zoneinfo import ZoneInfo
import requests
from pylatexenc.latex2text import LatexNodes2Text

from typing_extensions import TypedDict
from langchain_core.tools import BaseTool, tool
from langchain.chat_models import init_chat_model
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import (
    AIMessage,
    ToolMessage,
    HumanMessage,
    SystemMessage,
    RemoveMessage,
)
from image_storage import GeneratedImage, ImageStorageManager
from src.playwright_browser_toolkit_runner import (
    PLAYWRIGHT_BROWSER_TOOL_NAMES,
    PlaywrightBrowserThreadRunner,
)
from src.visual_crossing_weather import (
    VisualCrossingWeatherClient,
    VisualCrossingWeatherFormatter,
    VisualCrossingWeatherRequest,
)
from src.google_hotels_client import (
    GoogleHotelsClient,
    GoogleHotelsConsoleFormatter,
    GoogleHotelsRequest,
    sanitize_hotels_payload,
)
from src.google_flights_client import (
    GoogleFlightsClient,
    GoogleFlightsConsoleFormatter,
    GoogleFlightsRequest,
    sanitize_flights_payload,
)
from src.google_directions_client import GoogleDirectionsClient
from src.google_reverse_image_client import GoogleReverseImageClient
from src.google_reverse_image_tool import GoogleReverseImageTool, ReverseImageUploader
from src.google_lens_tool import GoogleLensClient, GoogleLensTool
from src.web_browser_tool import WebBrowserTool
from src.serper_image_search_tool import SerperImageSearchTool
from src.tavily_search_tool import RoutedTavilySearch
from src.anilist_client import AniListAPI, ANILIST_MEDIA_SORTS
from src.timer_reminder import TimerReminderManager
from src.asobi_ticket_agent import AsobiTicketQuery
from src.checkpoint_retention import (
    CheckpointRetentionError,
    RetainingPostgresSaver,
)
from src.context_token_counter import ContextTokenCounter, ContextTokenEstimate
from src.x_monitor import XMonitorToolError
from src.x_monitor_tool import (
    build_x_monitor_permission_failure,
    build_x_monitor_tool_failure,
    is_x_monitor_tool_user_allowed,
    list_x_monitor_tasks,
    send_x_link,
    start_x_monitor,
    stop_x_monitor,
)
from src.netease_music_tool import (
    DEFAULT_NETEASE_MUSIC_API_BASE,
    DEFAULT_ONEBOT_API_BASE,
    NeteaseMusicClient,
    NeteaseMusicToolError,
    OneBotMusicCardSender,
)

ANILIST_SORT_CHOICES_TEXT: str = ", ".join(ANILIST_MEDIA_SORTS)

# ---- 环境校验：仅在首次需要时检查，避免重复消耗 ----
_ENV_COMMON_CHECKED: bool = False
_ENV_OPENAI_CHECKED: bool = False
_ENV_GEMINI_CHECKED: bool = False


def _ensure_common_env_once() -> None:
    """
    进程级通用环境校验，仅首次调用时执行，确保运行环境可控。

    在容器场景下会放宽校验：检测到 `/.dockerenv` 或环境变量
    `SKIP_VENV_CHECK=1` 时，视为已满足隔离要求。

    Returns:
        None: 函数无返回值。

    Raises:
        AssertionError: 当未激活 `.venv` 且不在容器内时抛出。
    """
    global _ENV_COMMON_CHECKED
    if _ENV_COMMON_CHECKED:
        return
    in_docker = os.path.exists("/.dockerenv")
    skip_check = os.environ.get("SKIP_VENV_CHECK") == "1"
    assert (
        os.environ.get("VIRTUAL_ENV")
        or sys.prefix.endswith(".venv")
        or in_docker
        or skip_check
    ), "必须先激活虚拟环境 (.venv)。"
    _ENV_COMMON_CHECKED = True


def _ensure_openai_env_once() -> None:
    """
    OpenAI 相关环境校验，仅首次需要 OpenAI 时执行。

    Returns:
        None: 函数无返回值。

    Raises:
        AssertionError: 当缺少 `OPENAI_API_KEY` 环境变量时抛出。
    """
    global _ENV_OPENAI_CHECKED
    if _ENV_OPENAI_CHECKED:
        return
    assert os.environ.get("OPENAI_API_KEY"), "缺少 OPENAI_API_KEY 环境变量。"
    _ENV_OPENAI_CHECKED = True


# 说明：严禁在代码中硬编码密钥；请通过环境变量注入：


def _is_truthy_env(value: Optional[str]) -> bool:
    """
    判断环境变量字符串是否表示真值。

    Args:
        value (Optional[str]): 原始环境变量值。

    Returns:
        bool: 当值表示开启状态时返回 ``True``，否则返回 ``False``。

    Raises:
        AssertionError: 当 ``value`` 类型非法时抛出。
    """
    assert value is None or isinstance(value, str), "环境变量值必须为字符串或 None"
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _ensure_gemini_env_once() -> None:
    """
    Gemini 相关环境校验，仅首次需要 Gemini 时执行。

    允许以下任一环境变量组合通过：
    1. AI Studio / Gemini Developer API Key。
    2. Vertex AI 所需的基础环境变量。

    Returns:
        None: 函数无返回值。

    Raises:
        AssertionError: 当 AI Studio 与 Vertex AI 的环境变量都未配置时抛出。
    """
    global _ENV_GEMINI_CHECKED
    if _ENV_GEMINI_CHECKED:
        return
    ai_studio_ready = any(
        isinstance(os.environ.get(name), str) and os.environ.get(name, "").strip()
        for name in (
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_GENERATIVE_AI_API_KEY",
        )
    )
    vertex_ready = (
        _is_truthy_env(os.environ.get("GOOGLE_GENAI_USE_VERTEXAI"))
        and bool((os.environ.get("GOOGLE_CLOUD_PROJECT") or "").strip())
        and bool((os.environ.get("GOOGLE_CLOUD_LOCATION") or "").strip())
        and bool((os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or "").strip())
    )
    assert ai_studio_ready or vertex_ready, (
        "缺少 Gemini 可用环境变量。请设置 GOOGLE_API_KEY / GEMINI_API_KEY / "
        "GOOGLE_GENERATIVE_AI_API_KEY，或配置 GOOGLE_GENAI_USE_VERTEXAI=true、"
        "GOOGLE_CLOUD_PROJECT、GOOGLE_CLOUD_LOCATION、"
        "GOOGLE_APPLICATION_CREDENTIALS。"
    )
    _ENV_GEMINI_CHECKED = True


_BASE64_DATA_URL_PATTERN = re.compile(
    r"(data:[^;]+;base64,)[0-9A-Za-z+/=_-]+", re.IGNORECASE
)
_BASE64_SCHEME_PATTERN = re.compile(
    r"(base64://)[0-9A-Za-z+/=_-]+", re.IGNORECASE
)
_LATEX_MATH_PATTERN: re.Pattern[str] = re.compile(
    r"(?<!\\)\$\$(?P<block>.+?)(?<!\\)\$\$|(?<!\\)\$(?P<inline>.+?)(?<!\\)\$",
    re.DOTALL,
)
_LATEX_TO_UNICODE_CONVERTER = LatexNodes2Text()
CONTEXT_TRIGGER_TEXT_TOKENS_ENV: str = "CONTEXT_COMPRESS_TRIGGER_TEXT_TOKENS"
CONTEXT_KEEP_TEXT_TOKENS_ENV: str = "CONTEXT_KEEP_RECENT_TEXT_TOKENS"
CONTEXT_MIN_MESSAGES_ENV: str = "CONTEXT_MIN_KEEP_RECENT_MESSAGES"
CONTEXT_MAX_SUMMARY_TOKENS_ENV: str = "CONTEXT_MAX_SUMMARY_TEXT_TOKENS"
CONTEXT_MIN_COMPRESSIBLE_TOKENS_ENV: str = "CONTEXT_MIN_COMPRESSIBLE_TEXT_TOKENS"
CONTEXT_SUMMARY_MODEL_ENV: str = "CONTEXT_SUMMARY_MODEL"


def _convert_latex_to_unicode(text: str) -> str:
    """
    将文本中的 `$...$` 与 `$$...$$` 公式替换为 Unicode 文本。

    Args:
        text (str): 待处理字符串。

    Returns:
        str: 已完成公式替换的字符串。

    Raises:
        AssertionError: 当 ``text`` 不是字符串时抛出。
    """
    assert isinstance(text, str), "text 必须是字符串"

    def _replace(match: Match[str]) -> str:
        """
        将单个 LaTeX 数学片段转换为 Unicode 文本。

        Args:
            match (Match[str]): 公式匹配结果。

        Returns:
            str: 转换后的 Unicode 文本。

        Raises:
            AssertionError: 当匹配结果缺少公式内容时抛出。
        """
        latex_expr = match.group("block")
        if latex_expr is None:
            latex_expr = match.group("inline")
        assert latex_expr is not None, "匹配结果必须包含公式内容"
        # 使用数学模式解析，尽量保留公式中的语义与空格。
        converted = _LATEX_TO_UNICODE_CONVERTER.latex_to_text(f"${latex_expr}$")
        return converted.strip()

    return _LATEX_MATH_PATTERN.sub(_replace, text)


def _sanitize_for_logging(payload: object) -> str:
    """
    将待写入日志的对象转换为字符串，视频/图片等二进制段落替换为占位符。

    Args:
        payload (object): 待记录的消息对象，可以是字符串、列表或消息实例。

    Returns:
        str: 已将 base64 内容脱敏为 ``[BASE64...]`` 的字符串。

    Raises:
        None.
    """

    def _mask_binary(obj: object) -> object:
        """递归替换二进制字段，避免视频/图片原始数据写入日志。"""
        if isinstance(obj, (bytes, bytearray)):
            return f"[BINARY len={len(obj)}]"
        if isinstance(obj, dict):
            masked: dict[object, object] = {}
            for k, v in obj.items():
                if str(k).lower() == "data" and isinstance(v, (bytes, bytearray)):
                    masked[k] = f"[BINARY len={len(v)}]"
                else:
                    masked[k] = _mask_binary(v)
            return masked
        if isinstance(obj, (list, tuple)):
            return type(obj)(_mask_binary(x) for x in obj)
        return obj

    text = str(_mask_binary(payload))

    def _replace(match: Match[str]) -> str:
        """
        匹配到 base64 字符串后替换为占位符。

        Args:
            match (Match[str]): 正则匹配结果对象。

        Returns:
            str: 已替换 base64 内容后的子串。

        Raises:
            None.
        """
        return f"{match.group(1)}[BASE64...]"

    sanitized = _BASE64_DATA_URL_PATTERN.sub(_replace, text)
    sanitized = _BASE64_SCHEME_PATTERN.sub(
        lambda match: f"{match.group(1)}[BASE64...]", sanitized
    )
    return sanitized


def _sanitize_context_text(text: str) -> str:
    """
    将上下文统计与摘要提示中的大体积二进制文本替换为占位符。

    Args:
        text (str): 待清洗的文本。

    Returns:
        str: 已替换 base64 数据的文本。

    Raises:
        AssertionError: 当 text 不是字符串时抛出。
    """
    assert isinstance(text, str), "text 必须是字符串"
    sanitized = _BASE64_DATA_URL_PATTERN.sub(
        lambda match: f"{match.group(1)}[BASE64_OMITTED]", text
    )
    sanitized = _BASE64_SCHEME_PATTERN.sub(
        lambda match: f"{match.group(1)}[BASE64_OMITTED]", sanitized
    )
    return sanitized


def _apply_format(text: str) -> str:
    """
    根据环境变量 FORMAT 对文本执行统一格式化。

    Args:
        text (str): 原始字符串。

    Returns:
        str: FORMAT=1 时返回已去除 ``**`` 且公式已转 Unicode 的字符串；
        否则原样返回。
    """
    if os.environ.get("FORMAT") != "1":
        return text
    stripped = text.replace("**", "")
    return _convert_latex_to_unicode(stripped)


def _extract_text_content(message: Any) -> Optional[str]:
    """
    提取 LangChain v1 消息中的 type=text 内容，兼容旧版纯字符串结构。

    Args:
        message (Any): LangChain 消息对象或其 content 字段。

    Returns:
        Optional[str]: 抽取后的纯文本；若不存在文本块则返回 None。
    """

    if message is None:
        return None

    def _blocks_to_text(blocks: Sequence[Any]) -> str:
        texts: list[str] = []
        for block in blocks:
            text = ""
            if isinstance(block, dict):
                if (block.get("type") or "").lower() == "text":
                    text = block.get("text") or ""
            else:
                block_type = getattr(block, "type", "")
                if str(block_type).lower() == "text":
                    text = getattr(block, "text", "") or ""
            if text:
                texts.append(text)
        return "\n".join(texts)

    blocks = getattr(message, "content_blocks", None)
    if isinstance(blocks, Sequence) and blocks:
        text = _blocks_to_text(blocks)
        if text:
            return _apply_format(text)

    content = getattr(message, "content", None)
    if isinstance(content, str) and content:
        return _apply_format(content)
    if isinstance(content, Sequence) and not isinstance(
        content, (str, bytes, bytearray)
    ):
        text = _blocks_to_text(content)
        if text:
            return _apply_format(text)

    return None


def _normalize_blocked_ai_message(message: AIMessage) -> AIMessage:
    """
    将无文本内容且无工具调用的 AIMessage 归一化为可展示文本块。

    Args:
        message (AIMessage): 模型返回的 AI 消息对象。

    Returns:
        AIMessage: 若命中空回复场景，则返回带文本块的新消息；
            否则返回原消息对象。

    Raises:
        AssertionError: 当传入对象不是 ``AIMessage`` 实例时抛出。
    """

    assert isinstance(message, AIMessage), "message 必须为 AIMessage 实例"
    finish_reason = str(message.response_metadata.get("finish_reason") or "").upper()
    if message.content:
        return message
    if message.tool_calls or message.additional_kwargs.get("function_call"):
        return message
    reason = finish_reason or "UNKNOWN"
    if reason in {"PROHIBITED_CONTENT", "SAFETY"}:
        notice = f"（该请求触发安全策略，未返回内容：{reason}）"
    else:
        notice = f"（模型本次未返回可展示内容：{reason}）"
    return message.model_copy(
        update={
            "content": [
                {
                    "type": "text",
                    "text": notice,
                }
            ]
        }
    )


def _infer_model_provider(model_name: str) -> str:
    """
    推断模型提供方前缀。

    Args:
        model_name (str): LangChain 统一格式的模型名称，可选带前缀。

    Returns:
        str: 推断出的提供方前缀；无法判断时返回空字符串。

    Raises:
        AssertionError: 当模型名称为空字符串时抛出。
    """
    assert isinstance(model_name, str) and model_name.strip(), "model_name 不能为空"
    normalized = model_name.strip().lower()
    if ":" in normalized:
        return normalized.split(":", 1)[0]
    if normalized.startswith(("gpt", "gpt-4", "o1", "o3")):
        return "openai"
    if normalized.startswith("gemini"):
        return "google_genai"
    return ""


def _ensure_model_env_once(model_name: str) -> None:
    """
    根据模型名称触发相应的密钥校验逻辑。

    Args:
        model_name (str): LangChain 统一格式的模型名称。

    Returns:
        None: 函数无返回值。

    Raises:
        AssertionError: 当模型名称为空或缺失所需环境变量时抛出。
    """
    provider = _infer_model_provider(model_name)
    if provider == "fake":
        return
    if provider == "openai":
        _ensure_openai_env_once()
        return
    if provider.startswith("google"):
        if provider == "google_vertexai":
            return
        _ensure_gemini_env_once()
        return
    if provider == "gemini" or (
        provider == "" and model_name.strip().lower().startswith("gemini")
    ):
        _ensure_gemini_env_once()


def _cap_messages(prev: list | None, new: list | object) -> list:
    """基于内置 `add_messages` 的消息合并器。

    先使用 `add_messages(prev, new)` 完成标准的消息合并（与内置追加行为一致），
    再将新增消息内容追加到日志文件中，便于后续排查与回放。
    上下文截断与摘要由独立的 ``ContextCompressor`` 节点负责，避免在 reducer
    阶段丢失待压缩的旧消息。

    Args:
        prev (list|None): 既有消息列表。
        new (list|object): 新增消息（单条或列表）。

    Returns:
        list: 合并后的消息列表。
    """
    combined = add_messages(prev or [], new)

    log_dir = Path(os.environ.get("AGENT_MESSAGE_LOG_DIR", "logs")).expanduser()
    # 按日期分日志
    filename = time.strftime("%Y-%m-%d", time.localtime()) + ".log"
    log_path = log_dir / filename

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_text = _sanitize_for_logging(new)
        with log_path.open("a", encoding="utf-8") as fp:
            fp.write(f"{timestamp} | {log_text}\n")
    except Exception as err:
        sys.stderr.write(f"[CapMessages] 记录消息失败: {err}\n")

    return combined


def _read_positive_int_env(name: str, default: int) -> int:
    """
    读取正整数环境变量。

    Args:
        name (str): 环境变量名称。
        default (int): 未设置时使用的默认值。

    Returns:
        int: 环境变量指定的正整数或默认值。

    Raises:
        AssertionError: 当名称、默认值或环境变量值非法时抛出。
    """
    assert isinstance(name, str) and name.strip(), "环境变量名称不能为空"
    assert isinstance(default, int) and default > 0, "默认值必须为正整数"
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    assert raw.isdigit(), f"{name} 必须为正整数"
    value = int(raw)
    assert value > 0, f"{name} 必须大于 0"
    return value


@dataclass(frozen=True)
class ContextCompressionConfig:
    """
    群聊上下文压缩配置。

    Args:
        trigger_text_tokens (int): 触发压缩的上下文 token 阈值。
        keep_recent_text_tokens (int): 压缩后尽量保留的最近原文 token 数，
            包含文本、图片和视频。
        min_keep_recent_messages (int): 无论 token 数如何都至少保留的最近消息数；
            若消息数边界落在轮次中间，则向前扩展到完整轮次。
        max_summary_text_tokens (int): 群聊摘要允许的最大文本 token 数。
        min_compressible_text_tokens (int): 可压缩旧上下文至少达到该 token
            数时才调用摘要模型，包含文本、图片和视频。
        print_summary (bool): 是否在控制台输出压缩摘要正文。

    Returns:
        None: dataclass 初始化不返回额外值。

    Raises:
        AssertionError: 当参数不是正整数或阈值关系非法时抛出。
    """

    trigger_text_tokens: int = 35000
    keep_recent_text_tokens: int = 20000
    min_keep_recent_messages: int = 10
    max_summary_text_tokens: int = 3500
    min_compressible_text_tokens: int = 8000
    print_summary: bool = False

    def __post_init__(self) -> None:
        """
        校验压缩配置。

        Returns:
            None: 无返回值。

        Raises:
            AssertionError: 当配置值非法时抛出。
        """
        assert self.trigger_text_tokens > 0, "触发阈值必须为正整数"
        assert self.keep_recent_text_tokens > 0, "保留阈值必须为正整数"
        assert self.min_keep_recent_messages > 0, "最小保留消息数必须为正整数"
        assert self.max_summary_text_tokens > 0, "摘要 token 上限必须为正整数"
        assert (
            self.min_compressible_text_tokens > 0
        ), "可压缩区阈值必须为正整数"
        assert isinstance(self.print_summary, bool), "摘要输出开关必须为布尔值"
        assert (
            self.keep_recent_text_tokens < self.trigger_text_tokens
        ), "保留阈值必须小于触发阈值"

    @classmethod
    def from_env(cls) -> "ContextCompressionConfig":
        """
        从环境变量读取群聊上下文压缩配置。

        Returns:
            ContextCompressionConfig: 已校验的压缩配置。

        Raises:
            AssertionError: 当环境变量不是正整数或阈值关系非法时抛出。
        """
        return cls(
            trigger_text_tokens=_read_positive_int_env(
                CONTEXT_TRIGGER_TEXT_TOKENS_ENV, 35000
            ),
            keep_recent_text_tokens=_read_positive_int_env(
                CONTEXT_KEEP_TEXT_TOKENS_ENV, 20000
            ),
            min_keep_recent_messages=_read_positive_int_env(
                CONTEXT_MIN_MESSAGES_ENV, 10
            ),
            max_summary_text_tokens=_read_positive_int_env(
                CONTEXT_MAX_SUMMARY_TOKENS_ENV, 3500
            ),
            min_compressible_text_tokens=_read_positive_int_env(
                CONTEXT_MIN_COMPRESSIBLE_TOKENS_ENV, 8000
            ),
        )


@dataclass
class ContextTurn:
    """
    一段完整群聊轮次。

    Args:
        messages (list): 本轮包含的消息列表。
        text_tokens (int): 本轮文本与媒体的 token 估算总数。

    Returns:
        None: dataclass 初始化不返回额外值。

    Raises:
        AssertionError: 当消息列表为空或 token 数非法时抛出。
    """

    messages: list
    text_tokens: int

    def __post_init__(self) -> None:
        """
        校验轮次结构。

        Returns:
            None: 无返回值。

        Raises:
            AssertionError: 当轮次结构非法时抛出。
        """
        assert self.messages, "轮次消息不能为空"
        assert self.text_tokens >= 0, "轮次 token 数不能为负数"


class ContextCompressor:
    """
    面向群聊机器人的短期上下文压缩器。

    Args:
        config (ContextCompressionConfig): 压缩阈值配置。
        summary_model (Any): 具备 ``invoke`` 方法的摘要模型。
        token_encoder (Any | None): 可选 token 编码器；为空时使用 tiktoken。
        token_counter (ContextTokenCounter | None): 可选的共享上下文统计器。

    Returns:
        None: 类初始化不返回额外值。

    Raises:
        AssertionError: 当配置、摘要模型或 token 编码器不可用时抛出。
    """

    def __init__(
        self,
        config: ContextCompressionConfig,
        summary_model: Any,
        token_encoder: Any | None = None,
        token_counter: ContextTokenCounter | None = None,
    ) -> None:
        """
        初始化群聊上下文压缩器。

        Args:
            config (ContextCompressionConfig): 压缩阈值配置。
            summary_model (Any): 具备 ``invoke`` 方法的摘要模型。
            token_encoder (Any | None): 可选文本编码器。
            token_counter (ContextTokenCounter | None): 可选共享统计器。

        Returns:
            None: 无返回值。

        Raises:
            AssertionError: 当配置、摘要模型或统计器参数冲突时抛出。
        """
        assert isinstance(config, ContextCompressionConfig), "config 类型非法"
        invoke = getattr(summary_model, "invoke", None)
        assert callable(invoke), "summary_model 必须提供 invoke 方法"
        assert not (
            token_encoder is not None and token_counter is not None
        ), "token_encoder 与 token_counter 不能同时提供"
        self._config = config
        self._summary_model = summary_model
        self._token_counter = token_counter or ContextTokenCounter(
            text_sanitizer=_sanitize_context_text,
            token_encoder=token_encoder,
        )

    def count_text_tokens(self, text: str) -> int:
        """
        统计文本 token 数。

        Args:
            text (str): 待统计文本。

        Returns:
            int: 使用 cl100k_base 估算的 token 数。

        Raises:
            AssertionError: 当文本类型非法或编码失败时抛出。
        """
        return self._token_counter.count_text_tokens(text)

    def count_messages_tokens(self, messages: Sequence[Any]) -> int:
        """
        统计消息列表的文本、图片和视频 token。

        Args:
            messages (Sequence[Any]): LangChain 消息序列。

        Returns:
            int: 消息 token 总数。

        Raises:
            AssertionError: 当消息序列非法或 token 编码失败时抛出。
        """
        return self._token_counter.count_messages(messages).total_tokens

    def compress(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        根据当前 state 判断是否需要压缩，并返回 LangGraph state 更新。

        Args:
            state (dict[str, Any]): 当前 LangGraph state。

        Returns:
            dict[str, Any]: 需要写回 state 的更新；无需压缩时返回空字典。

        Raises:
            AssertionError: 当 state 结构非法、摘要失败或消息缺少 ID 时抛出。
        """
        assert isinstance(state, dict), "state 必须是字典"
        messages = list(state.get("messages") or [])
        summary = str(state.get("group_context_summary") or "").strip()
        total_tokens = self._token_counter.count_state(summary, messages).total_tokens
        if total_tokens <= self._config.trigger_text_tokens:
            return {}

        turns = self.split_turns(messages)
        total_message_count = sum(len(turn.messages) for turn in turns)
        if total_message_count <= self._config.min_keep_recent_messages:
            return {}

        keep_start = self.select_keep_start(turns)
        if keep_start <= 0:
            return {}

        compressed_messages = [
            message for turn in turns[:keep_start] for message in turn.messages
        ]
        assert compressed_messages, "被压缩消息不能为空"
        compressible_tokens = sum(turn.text_tokens for turn in turns[:keep_start])
        if compressible_tokens < self._config.min_compressible_text_tokens:
            return {}
        new_summary = self.summarize(summary, compressed_messages)
        summary_tokens = self.count_text_tokens(new_summary)
        last_message_id = str(getattr(compressed_messages[-1], "id", "") or "").strip()
        assert last_message_id, "被压缩消息缺少 ID"
        removals: list[RemoveMessage] = []
        for message in compressed_messages:
            message_id = str(getattr(message, "id", "") or "").strip()
            assert message_id, "被压缩消息缺少 ID"
            removals.append(RemoveMessage(id=message_id))

        old_round = state.get("compression_round") or 0
        assert isinstance(old_round, int) and old_round >= 0, "compression_round 非法"
        new_round = old_round + 1
        self.emit_compression_log(
            round_number=new_round,
            total_tokens_before=total_tokens,
            compressible_tokens=compressible_tokens,
            removed_messages=len(compressed_messages),
            kept_messages=sum(len(turn.messages) for turn in turns[keep_start:]),
            kept_turns=len(turns) - keep_start,
            summary_tokens=summary_tokens,
            summary=new_summary,
        )
        return {
            "group_context_summary": new_summary,
            "messages": removals,
            "compressed_until_message_id": last_message_id,
            "compression_round": new_round,
        }

    def emit_compression_log(
        self,
        round_number: int,
        total_tokens_before: int,
        compressible_tokens: int,
        removed_messages: int,
        kept_messages: int,
        kept_turns: int,
        summary_tokens: int,
        summary: str,
    ) -> None:
        """
        输出上下文压缩完成后的控制台日志。

        Args:
            round_number (int): 本次压缩后的累计轮次。
            total_tokens_before (int): 压缩触发前的上下文 token 总数。
            compressible_tokens (int): 本次被压缩旧上下文的 token 总数。
            removed_messages (int): 本次被移出原文上下文的消息数。
            kept_messages (int): 压缩后保留的最近消息数。
            kept_turns (int): 压缩后保留的最近完整轮次数。
            summary_tokens (int): 新摘要的文本 token 数。
            summary (str): 新生成的群聊上下文摘要。

        Returns:
            None: 无返回值。

        Raises:
            None: 日志输出不主动中断主流程。
        """
        print(
            "[ContextCompression] 已压缩群聊上下文: "
            f"round={round_number}, "
            f"total_tokens_before={total_tokens_before}, "
            f"compressible_tokens={compressible_tokens}, "
            f"removed_messages={removed_messages}, "
            f"kept_messages={kept_messages}, "
            f"kept_turns={kept_turns}, "
            f"summary_tokens={summary_tokens}",
            flush=True,
        )
        if self._config.print_summary:
            print(
                "[ContextCompression] 摘要正文:\n"
                f"{_sanitize_context_text(summary).strip()}",
                flush=True,
            )

    def split_turns(self, messages: Sequence[Any]) -> list[ContextTurn]:
        """
        按 HumanMessage 切分完整群聊轮次。

        Args:
            messages (Sequence[Any]): LangChain 消息序列。

        Returns:
            list[ContextTurn]: 已切分的完整轮次列表。

        Raises:
            AssertionError: 当 messages 不是序列时抛出。
        """
        assert isinstance(messages, Sequence), "messages 必须是序列"
        turns: list[ContextTurn] = []
        current: list[Any] = []
        for message in messages:
            if isinstance(message, HumanMessage) and current:
                turns.append(
                    ContextTurn(
                        messages=current,
                        text_tokens=self.count_messages_tokens(current),
                    )
                )
                current = [message]
            else:
                current.append(message)
        if current:
            turns.append(
                ContextTurn(
                    messages=current,
                    text_tokens=self.count_messages_tokens(current),
                )
            )
        return turns

    def select_keep_start(self, turns: Sequence[ContextTurn]) -> int:
        """
        选择压缩后最近原文轮次的起始下标。

        保留区按最近消息数兜底，并向前扩展到完整轮次；超过消息数兜底后，
        继续在最近原文 token 预算内尽量多保留旧轮次。

        Args:
            turns (Sequence[ContextTurn]): 已按时间排序的轮次。

        Returns:
            int: 需要保留的第一轮下标。

        Raises:
            AssertionError: 当 turns 不是序列时抛出。
        """
        assert isinstance(turns, Sequence), "turns 必须是序列"
        keep_start = len(turns)
        kept_tokens = 0
        kept_messages = 0
        for index in range(len(turns) - 1, -1, -1):
            turn = turns[index]
            required_by_min_messages = (
                kept_messages < self._config.min_keep_recent_messages
            )
            within_token_budget = (
                kept_tokens + turn.text_tokens <= self._config.keep_recent_text_tokens
            )
            if required_by_min_messages or within_token_budget:
                keep_start = index
                kept_tokens += turn.text_tokens
                kept_messages += len(turn.messages)
                continue
            break
        return keep_start

    def summarize(self, previous_summary: str, messages: Sequence[Any]) -> str:
        """
        将旧摘要与待压缩消息合并成新的群聊语境摘要。

        Args:
            previous_summary (str): 上一版群聊上下文摘要。
            messages (Sequence[Any]): 本次需要压缩的旧消息。

        Returns:
            str: 新的群聊上下文摘要。

        Raises:
            AssertionError: 当输入非法、模型无输出或摘要超出上限时抛出。
        """
        assert isinstance(previous_summary, str), "previous_summary 必须是字符串"
        assert isinstance(messages, Sequence) and messages, "messages 不能为空"
        prompt = self.build_summary_prompt(previous_summary, messages)
        response = self._summary_model.invoke([HumanMessage(content=prompt)])
        summary = self.extract_response_text(response)
        assert summary.strip(), "上下文摘要模型返回为空"
        token_count = self.count_text_tokens(summary)
        assert (
            token_count <= self._config.max_summary_text_tokens
        ), "上下文摘要超过 token 上限"
        return summary.strip()

    def build_summary_prompt(
        self, previous_summary: str, messages: Sequence[Any]
    ) -> str:
        """
        构造群聊语境摘要 prompt。

        Args:
            previous_summary (str): 上一版群聊上下文摘要。
            messages (Sequence[Any]): 本次需要压缩的旧消息。

        Returns:
            str: 摘要模型使用的 prompt。

        Raises:
            AssertionError: 当输入类型非法时抛出。
        """
        assert isinstance(previous_summary, str), "previous_summary 必须是字符串"
        assert isinstance(messages, Sequence), "messages 必须是序列"
        message_text = "\n\n".join(
            self.message_to_summary_text(message) for message in messages
        )
        if previous_summary.strip():
            previous_block = previous_summary.strip()
        else:
            previous_block = "无"
        return (
            "你在为一个群聊角色扮演Agent滚动压缩短期上下文。\n"
            "请把上一版摘要视为更早的群聊经过，把本次旧消息视为随后发生的新内容，"
            "合并更新为一份连续摘要；不要只总结最新消息。\n"
            "摘要主体是事件经过：按时间顺序保留主要话题怎样出现、谁怎样回应、"
            "事情怎样变化、得到什么结果以及停在哪里。"
            "在 token 预算充足时，不要仅因出现新话题就删除上一版摘要中的主要事件；"
            "重复事件应合并，久远且不再影响接话的普通闲聊可逐渐压缩成一句。\n"
            "原有分类栏目用于补充仍影响后续接话的语境，不是数据库字段。"
            "只写消息中确有依据且仍然有用的信息；没有信息写“无”，"
            "不要为了填满栏目生成空泛套话，也不要在多个栏目机械重复同一事实。\n"
            "必须适当保留重要的 user_id、user_name、关键发言、情绪、梗、"
            "用户明确指出的称呼偏好、关系、明确偏好或雷点。\n"
            "保持发言者、事实主体与行为归属准确。"
            "不要把一次发言推断成稳定倾向、偏好或关系；"
            "参与者倾向、特殊关系和明确偏好只有消息明确说明时才能记录。"
            "单次发言只记录本次观点或贡献，禁止补充人格评价。"
            "不能从互动顺畅或争执后和好推断特殊关系。\n"
            "工具信息只有在会影响后续群聊接话、用户可能追问、"
            "或Agent已经基于它做出判断时才保留；"
            "保留工具名、调用目的、关键参数、关键结果，"
            "不要保留原始 JSON、HTML 或大段中间结果。\n"
            "图片和视频只保留数量、文件名、当时结论或上下文，不要保留 base64。\n"
            "Agent自己角色的设定已经在其它地方说明了，"
            "不要在摘要中提到或猜测Agent扮演的角色。\n"
            f"输出摘要不超过 {self._config.max_summary_text_tokens} 个 token。\n"
            "请使用以下固定结构：\n"
            "被压缩上下文的话题：\n"
            "被压缩上下文的主要经过：\n"
            "被压缩上下文的氛围：\n"
            "参与者与发言倾向：\n"
            "群内梗、特殊关系：\n"
            "明确偏好或雷点：\n"
            "最近图片/视频语境：\n"
            "工具调用要点：\n"
            "Agent最近说过什么：\n"
            "需要延续或避免重复的点：\n\n"
            f"上一版摘要：\n{previous_block}\n\n"
            f"需要压缩的旧消息：\n{message_text}"
        )

    def message_to_summary_text(self, message: Any) -> str:
        """
        将单条消息转换为可供摘要模型使用的安全文本。

        Args:
            message (Any): LangChain 消息对象或兼容对象。

        Returns:
            str: 不含原始 base64 的摘要输入文本。

        Raises:
            None: 本方法对未知消息类型使用字符串表示。
        """
        role = self.message_role(message)
        parts = [f"[{role}]"]
        content = getattr(message, "content", None)
        content_text = self.content_to_safe_text(content)
        if content_text:
            parts.append(content_text)
        if isinstance(message, AIMessage) and message.tool_calls:
            calls = _sanitize_context_text(
                json.dumps(message.tool_calls, ensure_ascii=False, default=str)
            )
            parts.append(f"工具调用: {calls}")
        if isinstance(message, ToolMessage):
            tool_name = str(getattr(message, "name", "") or "").strip()
            if tool_name:
                parts.append(f"工具名: {tool_name}")
        if len(parts) == 1:
            parts.append(_sanitize_context_text(str(message)))
        return "\n".join(parts)

    @staticmethod
    def message_role(message: Any) -> str:
        """
        获取消息角色名称。

        Args:
            message (Any): LangChain 消息对象。

        Returns:
            str: 用于摘要输入的中文角色名称。

        Raises:
            None: 未知类型返回 ``消息``。
        """
        if isinstance(message, HumanMessage):
            return "用户"
        if isinstance(message, AIMessage):
            return "Agent"
        if isinstance(message, ToolMessage):
            return "工具"
        if isinstance(message, SystemMessage):
            return "系统"
        return "消息"

    def content_to_safe_text(self, content: Any) -> str:
        """
        将消息 content 转换为不含媒体原始数据的摘要文本。

        Args:
            content (Any): 消息 content 字段。

        Returns:
            str: 可用于摘要 Prompt 的安全文本。

        Raises:
            None: 未知结构使用脱敏后的字符串表示。
        """
        if content is None:
            return ""
        if isinstance(content, str):
            return _sanitize_context_text(_apply_format(content))
        if isinstance(content, (bytes, bytearray)):
            return f"[二进制内容已省略 len={len(content)}]"
        if isinstance(content, Sequence) and not isinstance(
            content, (str, bytes, bytearray)
        ):
            blocks: list[str] = []
            for block in content:
                blocks.append(self.block_to_safe_text(block))
            return "\n".join(item for item in blocks if item)
        return _sanitize_context_text(str(content))

    def block_to_safe_text(self, block: Any) -> str:
        """
        将多模态 content block 转换为安全文本。

        Args:
            block (Any): content block。

        Returns:
            str: 安全文本或媒体占位符。

        Raises:
            None: 未知结构使用脱敏后的字符串表示。
        """
        if isinstance(block, dict):
            block_type = str(block.get("type") or "").lower()
            if block_type == "text":
                return _sanitize_context_text(
                    _apply_format(str(block.get("text") or ""))
                )
            if block_type == "image_url":
                return "[图片已省略]"
            if block_type in {"media", "video"}:
                mime = str(block.get("mime_type") or "").strip()
                return f"[媒体已省略 mime={mime or 'unknown'}]"
            sanitized = dict(block)
            if "data" in sanitized:
                sanitized["data"] = "[BINARY_OMITTED]"
            return _sanitize_context_text(
                json.dumps(sanitized, ensure_ascii=False, default=str)
            )
        block_type = str(getattr(block, "type", "") or "").lower()
        if block_type == "text":
            return _sanitize_context_text(
                _apply_format(str(getattr(block, "text", "") or ""))
            )
        if block_type == "image_url":
            return "[图片已省略]"
        return _sanitize_context_text(str(block))

    @staticmethod
    def extract_response_text(response: Any) -> str:
        """
        从摘要模型响应中提取文本。

        Args:
            response (Any): 模型响应对象。

        Returns:
            str: 提取到的文本。

        Raises:
            AssertionError: 当响应无法转换为文本时抛出。
        """
        if isinstance(response, str):
            return _sanitize_context_text(response)
        text = _extract_text_content(response)
        if text:
            return _sanitize_context_text(text)
        content = getattr(response, "content", None)
        if isinstance(content, str):
            return _sanitize_context_text(content)
        raise AssertionError("无法从上下文摘要模型响应中提取文本")


@dataclass
class RapidAPIHotelSearchClient:
    """
    RapidAPI 酒店搜索客户端。

    通过 RapidAPI 上的 TripAdvisor Scraper（pradeepbardiya13）接口，
    根据城市关键词检索热门酒店信息。
    """

    timeout: int = 15

    def __post_init__(self) -> None:
        """
        初始化客户端并校验必要的环境变量。

        Raises:
            AssertionError: 当缺少 `RAPIDAPI_KEY` 环境变量时抛出。
        """

        api_key = os.environ.get("RAPIDAPI_KEY")
        assert api_key, "缺少 RAPIDAPI_KEY 环境变量。"
        host_raw = os.environ.get("RAPIDAPI_HOTEL_HOST")
        host = (host_raw or "tripadvisor-scraper.p.rapidapi.com").strip()
        assert host, "RAPIDAPI_HOTEL_HOST 配置为空，请提供有效主机名。"
        self._api_key = api_key
        self._api_host = host
        self._base_url = f"https://{host}"

    def _get(self, path: str, params: dict[str, str]) -> dict:
        """
        执行 GET 请求并返回 JSON。

        Args:
            path (str): 接口路径（以斜杠开头）。
            params (dict[str, str]): 查询参数键值对。

        Returns:
            dict: 解析后的 JSON 响应。

        Raises:
            AssertionError: 当 path 非法时抛出。
            ValueError: 当网络请求失败或响应格式异常时抛出。
        """

        assert isinstance(path, str) and path.startswith("/"), "path 必须以 / 开头"
        url = f"{self._base_url}{path}"
        headers = {
            "X-RapidAPI-Key": self._api_key,
            "X-RapidAPI-Host": self._api_host,
        }
        response = requests.get(
            url, headers=headers, params=params, timeout=self.timeout
        )
        if response.status_code == 401:
            raise ValueError(
                "RapidAPI 鉴权失败：请确认 RAPIDAPI_KEY 是否正确且已订阅相关 API。"
            )
        if response.status_code == 403:
            raise ValueError(
                "RapidAPI 拒绝访问：请检查是否完成 API 订阅、是否启用了请求 IP，或是否触发配额限制。"
            )
        if response.status_code != 200:
            raise ValueError(f"调用 {path} 失败，HTTP {response.status_code}")
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} 返回非 JSON 数据") from exc

    def search_hotels(self, city: str, limit: int) -> list[dict[str, str]]:
        """
        按城市查询酒店列表。

        Args:
            city (str): 城市关键词，需为非空字符串。
            limit (int): 返回的酒店数量上限，取值范围 1-10。

        Returns:
            list[dict[str, str]]: 酒店信息列表。

        Raises:
            AssertionError: 当参数类型或取值非法时抛出。
            ValueError: 当请求失败或返回数据格式异常时抛出。
        """

        assert isinstance(city, str) and city.strip(), "city 必须为非空字符串"
        assert (
            isinstance(limit, int) and 1 <= limit <= 10
        ), "limit 必须为 1 到 10 的整数"

        payload = self._get(
            "/hotels/list",
            {
                "query": city.strip(),
                "page": "1",
            },
        )

        hotel_items = payload.get("results")
        if not isinstance(hotel_items, list):
            raise ValueError("酒店检索返回数据格式异常")

        results: list[dict[str, str]] = []
        for item in hotel_items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("title") or item.get("name") or "").strip()
            if not name:
                continue
            rating_val = item.get("rating")
            rating_text = str(rating_val) if rating_val else "暂无评分"
            price_info = item.get("price_range_usd") or {}
            if isinstance(price_info, dict) and (
                price_info.get("min") or price_info.get("max")
            ):
                min_price = price_info.get("min")
                max_price = price_info.get("max")
                if min_price and max_price:
                    price_text = f"${min_price} - ${max_price}"
                elif min_price:
                    price_text = f"低至 ${min_price}"
                elif max_price:
                    price_text = f"最高 ${max_price}"
                else:
                    price_text = "暂无价格"
            else:
                price_text = "暂无价格"
            address = str(item.get("address") or "").strip()
            if not address:
                detailed = item.get("detailed_address")
                if isinstance(detailed, dict):
                    address_parts = [
                        detailed.get("street"),
                        detailed.get("city"),
                        detailed.get("state"),
                        detailed.get("country"),
                    ]
                    address = ", ".join(
                        part.strip()
                        for part in address_parts
                        if isinstance(part, str) and part.strip()
                    )
            if not address:
                address = f"{city.strip()}（地址未提供）"
            results.append(
                {
                    "name": name,
                    "address": address,
                    "rating": rating_text,
                    "price": price_text,
                }
            )
            if len(results) >= limit:
                break

        if not results:
            raise ValueError("未查询到符合条件的酒店，请调整关键词。")

        return results


class State(TypedDict, total=False):
    """Agent 的图状态。"""

    messages: Annotated[list, _cap_messages]
    group_context_summary: str
    compressed_until_message_id: str
    compression_round: int


@tool("xmonitor")
def xmonitor(
    action: str,
    group_id: int,
    user_id: int,
    username: str = "",
    interval_seconds: float = 300,
) -> str:
    """
    管理指定群中的 X 账号推文监控任务。

    必须使用当前用户消息中提供的 Group_id 和 User_id，不得猜测或修改。

    Args:
        action (str): 操作类型，仅支持 ``start``、``stop``、``list``。
        group_id (int): 当前用户消息中的 Group_id。
        user_id (int): 当前用户消息中的 User_id。
        username (str): X 账号 handle，不是显示名称。start 和 stop 时必填。
            示例：``kana_hanaiwa`` 或 ``@kana_hanaiwa``，不要传入
            ``Kana Hanaiwa``。
        interval_seconds (float): 轮询间隔秒数。start 时可省略，默认 300 秒。

    Returns:
        str: JSON 字符串，包含操作结果或结构化失败。

    Raises:
        None: 工具边界会将所有运行异常转换为失败 JSON。
    """
    normalized_action = action.strip().lower()
    try:
        assert normalized_action in {"start", "stop", "list"}, (
            "action 仅支持 start/stop/list"
        )
        assert group_id > 0, "group_id 必须为正整数"
        assert user_id > 0, "user_id 必须为正整数"
        result: dict[str, object]
        if not is_x_monitor_tool_user_allowed(user_id):
            result = build_x_monitor_permission_failure(
                normalized_action,
                group_id,
                user_id,
            )
        elif normalized_action == "start":
            assert username.strip(), "start 操作必须提供 username"
            assert interval_seconds > 0, "interval_seconds 必须大于 0"
            start_x_monitor(
                username=username,
                interval_seconds=interval_seconds,
                group_id=group_id,
                user_id=user_id,
            )
            result = {
                "action": normalized_action,
                "group_id": group_id,
                "user_id": user_id,
                "username": username.lstrip("@"),
                "interval_seconds": interval_seconds,
                "status": "started",
            }
        elif normalized_action == "stop":
            assert username.strip(), "stop 操作必须提供 username"
            stopped = stop_x_monitor(username=username, group_id=group_id)
            result = {
                "action": normalized_action,
                "group_id": group_id,
                "user_id": user_id,
                "username": username.lstrip("@"),
                "stopped": stopped,
            }
        else:
            result = {
                "action": normalized_action,
                "group_id": group_id,
                "user_id": user_id,
                "tasks": list_x_monitor_tasks(group_id=group_id),
            }
    except XMonitorToolError as exc:
        result = build_x_monitor_tool_failure(
            action=normalized_action,
            group_id=group_id,
            user_id=user_id,
            error=exc,
            username=username,
        )
    except AssertionError as exc:
        result = {
            "action": normalized_action,
            "group_id": group_id,
            "user_id": user_id,
            "status": "failed",
            "error": "invalid_argument",
            "message": str(exc),
        }
        if username.strip():
            result["username"] = username.strip().lstrip("@")
    except Exception as exc:
        print(
            f"[XMonitor Tool Error] {type(exc).__name__}: {exc}",
            flush=True,
        )
        result = {
            "action": normalized_action,
            "group_id": group_id,
            "user_id": user_id,
            "status": "failed",
            "error": "unknown_error",
            "message": "xmonitor 执行失败，原因未知。",
        }
        if username.strip():
            result["username"] = username.strip().lstrip("@")
    output = json.dumps(result, ensure_ascii=False)
    print(
        f"\033[94m{time.strftime('[%m-%d %H:%M:%S]', time.localtime())}\033[0m [XMonitor Tool Output] {output}",
        flush=True,
    )
    return output


@tool("xlink")
def xlink(url: str, group_id: int, user_id: int) -> str:
    """
    拉取指定 X/Twitter 推文链接，并将推文截图发送到当前群。

    必须使用当前用户消息中提供的 Group_id 和 User_id，不得猜测或修改。

    Args:
        url (str): X/Twitter 推文链接。
        group_id (int): 当前用户消息中的 Group_id。
        user_id (int): 当前用户消息中的 User_id。

    Returns:
        str: 已发送推文的正文。

    Raises:
        AssertionError: 当输入参数或推文链接非法时抛出。
        RuntimeError: 当 X API 或 OneBot 请求失败时抛出。
        ValueError: 当 X API 返回字段格式非法时抛出。
    """
    assert url.strip(), "url 不能为空"
    assert group_id > 0, "group_id 必须为正整数"
    assert user_id > 0, "user_id 必须为正整数"
    item = send_x_link(url=url, group_id=group_id, user_id=user_id)
    print(
        f"\033[94m{time.strftime('[%m-%d %H:%M:%S]', time.localtime())}\033[0m [XLink Tool Output] {item.text}",
        flush=True,
    )
    return item.text


@tool("netease_music_search")
def netease_music_search(keyword: str, limit: int = 5) -> str:
    """
    搜索网易云音乐单曲并返回可用于发送音乐卡片的歌曲 ID 候选。

    只负责搜索，不会发送消息。搜索词建议包含歌名和歌手名，例如
    ``海阔天空 Beyond``。不要猜测歌曲 ID，应从本工具结果中选择。

    Args:
        keyword (str): 歌名或“歌名 + 歌手”关键词。
        limit (int): 返回候选数量，范围为 1 至 5，默认 5。

    Returns:
        str: JSON 字符串，包含搜索状态和歌曲候选列表。

    Raises:
        AssertionError: 当关键词为空、过长或 limit 超出范围时抛出。
    """
    normalized_keyword = keyword.strip()
    assert normalized_keyword, "keyword 不能为空"
    assert len(normalized_keyword) <= 100, "keyword 长度不能超过 100"
    assert 1 <= limit <= 5, "limit 必须在 1 到 5 之间"

    api_base = os.environ.get(
        "NETEASE_MUSIC_API_BASE",
        DEFAULT_NETEASE_MUSIC_API_BASE,
    ).strip()
    assert api_base, "NETEASE_MUSIC_API_BASE 不能为空"
    client = NeteaseMusicClient(base_url=api_base)
    try:
        songs = client.search(normalized_keyword, limit=limit)
        if songs:
            result: dict[str, object] = {
                "status": "success",
                "query": normalized_keyword,
                "source": "netease",
                "count": len(songs),
                "songs": [
                    song.to_dict(rank)
                    for rank, song in enumerate(songs, start=1)
                ],
            }
        else:
            result = {
                "status": "not_found",
                "query": normalized_keyword,
                "source": "netease",
                "count": 0,
                "songs": [],
                "message": "未找到相关网易云歌曲，请补充歌手名或调整关键词。",
            }
    except NeteaseMusicToolError as exc:
        result = {
            "status": "failed",
            "query": normalized_keyword,
            "source": "netease",
            "error": exc.error_code,
            "message": str(exc),
        }
    except AssertionError as exc:
        result = {
            "status": "failed",
            "query": normalized_keyword,
            "source": "netease",
            "error": "invalid_response",
            "message": str(exc),
        }
    output = json.dumps(result, ensure_ascii=False)
    print(
        f"\033[94m{time.strftime('[%m-%d %H:%M:%S]', time.localtime())}\033[0m "
        f"[NeteaseMusicSearch Tool Output] {output}",
        flush=True,
    )
    return output


@tool("send_netease_music_card")
def send_netease_music_card(song_id: str, group_id: int) -> str:
    """
    把指定网易云歌曲 ID 作为音乐卡片发送到目标 QQ 群。

    ``song_id`` 必须来自 ``netease_music_search`` 的搜索结果；
    ``group_id`` 由调用时直接提供，不进行上下文提取。

    Args:
        song_id (str): 网易云单曲 ID，例如 ``1357375695``。
        group_id (int): 接收音乐卡片的目标 QQ 群号。

    Returns:
        str: JSON 字符串，包含发送状态和 OneBot 消息 ID。

    Raises:
        AssertionError: 当 song_id 或 group_id 非法时抛出。
    """
    normalized_song_id = song_id.strip()
    assert normalized_song_id.isdigit(), "song_id 必须是数字字符串"
    assert group_id > 0, "group_id 必须为正整数"

    api_base = os.environ.get("ONEBOT_API_BASE", DEFAULT_ONEBOT_API_BASE).strip()
    access_token = os.environ.get("ONEBOT_ACCESS_TOKEN", "").strip()
    assert api_base, "ONEBOT_API_BASE 不能为空"
    sender = OneBotMusicCardSender(
        api_base=api_base,
        access_token=access_token,
    )
    try:
        message_id = sender.send(normalized_song_id, group_id)
        result: dict[str, object] = {
            "status": "sent",
            "action": "send_netease_music_card",
            "song_id": normalized_song_id,
            "group_id": group_id,
            "message_id": message_id,
        }
    except NeteaseMusicToolError as exc:
        result = {
            "status": "failed",
            "action": "send_netease_music_card",
            "song_id": normalized_song_id,
            "group_id": group_id,
            "error": exc.error_code,
            "message": str(exc),
        }
    except AssertionError as exc:
        result = {
            "status": "failed",
            "action": "send_netease_music_card",
            "song_id": normalized_song_id,
            "group_id": group_id,
            "error": "invalid_response",
            "message": str(exc),
        }
    output = json.dumps(result, ensure_ascii=False)
    print(
        f"\033[94m{time.strftime('[%m-%d %H:%M:%S]', time.localtime())}\033[0m "
        f"[NeteaseMusicCard Tool Output] {output}",
        flush=True,
    )
    return output


@dataclass
class AgentConfig:
    """Agent 运行配置。"""

    model_name: str = "openai:gpt-4o-mini"
    pg_conn: str = ""
    thread_id: str = "demo-plus"
    use_memory_ckpt: bool = False
    enable_tools: bool = False
    checkpoint_retention_limit: int = 5
    # 用于持久记忆（langmem）命名空间的 store 隔离标识，由环境变量 STORE_ID 注入
    store_id: str = ""


class SQLCheckpointAgentStreamingPlus:
    """多轮工具 + 强化综合 的流式 Agent。"""

    def __init__(
        self,
        config: AgentConfig,
        reminder_manager: TimerReminderManager | None = None,
    ) -> None:
        # 仅首次进行通用环境校验
        _ensure_common_env_once()

        dry_run = os.environ.get("DRY_RUN") == "1"
        self._config = config
        if dry_run:
            self._config.use_memory_ckpt = True

        if not self._config.use_memory_ckpt:
            assert self._config.pg_conn, "必须通过 LANGGRAPH_PG 提供 Postgres 连接串。"
            assert (
                self._config.checkpoint_retention_limit >= 1
            ), "checkpoint_retention_limit 必须大于等于 1"
        if self._config.model_name != "fake:echo":
            _ensure_model_env_once(self._config.model_name)
        os.environ["IMAGE_PROVIDER"] = "gemini"

        env_tools = os.environ.get("ENABLE_TOOLS")
        if env_tools is None:
            self._enable_tools = config.enable_tools
        else:
            self._enable_tools = env_tools in {"1", "true", "True"}

        # 预先读取并缓存系统提示内容（从外部文件），避免每轮重复IO
        self._sys_msg_content: str = self._load_sys_msg_content()
        self._reminder_manager = reminder_manager
        self._asobi_query = AsobiTicketQuery()
        self._image_manager: Optional[ImageStorageManager] = None
        self._generated_images: list[GeneratedImage] = []
        self._playwright_runner = PlaywrightBrowserThreadRunner()
        self._context_token_counter = ContextTokenCounter(
            text_sanitizer=_sanitize_context_text
        )
        self._context_compressor = self._build_context_compressor()

        self._graph = self._build_graph()
        self._printed_in_round: bool = False
        # 当前持久记忆命名空间（供 langmem 工具使用）；由外部在请求前设置
        self._memory_namespace: str = ""

    def shutdown(self) -> None:
        """
        释放 Agent 自身持有的 Playwright 浏览器。

        Returns:
            None: 函数无返回值。

        Raises:
            None.
        """
        if isinstance(self._playwright_runner, PlaywrightBrowserThreadRunner):
            self._playwright_runner.close()

    def set_memory_namespace(self, namespace: str) -> None:
        """
        设置当前会话关联的持久记忆命名空间。

        Args:
            namespace (str): 命名空间字符串，应确保全局唯一且可追踪。

        Raises:
            AssertionError: 当 namespace 非法时抛出。
        """
        assert isinstance(namespace, str) and namespace.strip(), "namespace 不能为空"
        self._memory_namespace = namespace.strip()

    def set_image_manager(self, manager: ImageStorageManager) -> None:
        """
        设置图像存储管理器，供多模态与生成工具使用。

        Args:
            manager (ImageStorageManager): 已初始化的图像管理器实例。

        Raises:
            AssertionError: 当传入对象类型不匹配时抛出。
        """
        assert isinstance(manager, ImageStorageManager), "manager 类型无效"
        self._image_manager = manager

    def consume_generated_images(self) -> list[GeneratedImage]:
        """
        读取并清空最近一次会话生成的图像列表。

        Returns:
            list[GeneratedImage]: 生成图像集合，按生成顺序排列。
        """
        images = list(self._generated_images)
        self._generated_images = []
        return images

    def _require_image_manager(self) -> ImageStorageManager:
        """
        获取已配置的图像管理器。

        Returns:
            ImageStorageManager: 图像存储管理器实例。

        Raises:
            AssertionError: 当图像管理器尚未设置时抛出。
        """
        if not isinstance(self._image_manager, ImageStorageManager):
            raise AssertionError("图像管理器尚未配置")
        return self._image_manager

    def _load_sys_msg_content(self) -> str:
        """读取系统提示词内容。

        优先从环境变量 `SYS_MSG_FILE` 指定的路径读取系统提示文本；
        要求该文件必须存在且非空，否则抛出断言错误。

        Returns:
            str: 系统提示词全文内容。

        Raises:
            AssertionError: 当环境变量未设置、文件不存在或内容为空时抛出。
        """
        path = os.environ.get("SYS_MSG_FILE")
        assert path, "必须通过环境变量 SYS_MSG_FILE 指定系统提示文件路径。"
        abs_path = os.path.abspath(path)
        assert os.path.isfile(abs_path), f"系统提示文件不存在: {abs_path}"
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert content and content.strip(), "系统提示文件内容为空。"
        return content

    @property
    def system_prompt_character_count(self) -> int:
        """返回当前已加载系统 Prompt 的字符数量。

        Returns:
            int: 系统 Prompt 的字符数量。

        Raises:
            AssertionError: 当系统 Prompt 尚未加载或内容为空时抛出。
        """
        assert self._sys_msg_content, "系统 Prompt 尚未加载"
        return len(self._sys_msg_content)

    def _build_playwright_browser_tools(self) -> list[BaseTool]:
        """
        构造 Playwright Browser Toolkit 的同步浏览器工具。

        Returns:
            list[BaseTool]: 可注册到 Agent 的 Playwright 浏览器工具列表。

        Raises:
            AssertionError: 当 Toolkit 返回的工具集合不符合预期时抛出。
        """
        assert isinstance(
            self._playwright_runner, PlaywrightBrowserThreadRunner
        ), "Playwright runner 未初始化"
        playwright_tools = self._playwright_runner.build_tools()
        tool_names = {str(playwright_tool.name) for playwright_tool in playwright_tools}
        assert tool_names == PLAYWRIGHT_BROWSER_TOOL_NAMES, (
            "Playwright Browser Toolkit 工具集合不符合预期: "
            f"{sorted(tool_names)}"
        )
        return playwright_tools

    def _build_context_compressor(self) -> ContextCompressor:
        """
        构建群聊上下文压缩器。

        Returns:
            ContextCompressor: 已配置的短期上下文压缩器。

        Raises:
            AssertionError: 当环境变量或摘要模型配置非法时抛出。
        """
        config = ContextCompressionConfig.from_env()
        summary_model = self._build_context_summary_model()
        return ContextCompressor(
            config=config,
            summary_model=summary_model,
            token_counter=self._context_token_counter,
        )

    def _build_context_summary_model(self) -> Any:
        """
        构建上下文摘要模型。

        ``CONTEXT_SUMMARY_MODEL`` 有值时使用该模型；否则沿用当前 Agent 模型。
        本方法不做模型回退，配置错误会显式抛出断言或初始化异常。

        Returns:
            Any: 具备 ``invoke`` 方法的摘要模型。

        Raises:
            AssertionError: 当模型名为空或缺少必要环境变量时抛出。
        """
        model_name = os.environ.get(CONTEXT_SUMMARY_MODEL_ENV, "").strip()
        if not model_name:
            model_name = self._config.model_name
        assert model_name.strip(), "上下文摘要模型名称不能为空"
        if model_name in {"fake:echo", "fake:summary"}:
            return _FakeContextSummaryModel()
        _ensure_model_env_once(model_name)
        return init_chat_model(model_name)

    def _build_graph(self):
        model_name = self._config.model_name
        if model_name == "fake:echo":
            llm = _FakeStreamingEcho()
            tools: list = []
            llm_tools_auto = llm
            llm_tools_none = llm
        else:
            llm = init_chat_model(model_name)
            tools = []
            if self._enable_tools:
                if os.environ.get("TAVILY_API_KEY"):
                    tools = [RoutedTavilySearch(max_results=5)]

                serper_api_key = os.environ.get("SERPER_API_KEY", "").strip()
                if serper_api_key:
                    tools.append(SerperImageSearchTool(api_key=serper_api_key))

                enable_legacy_web_browser_tool: bool = False
                if enable_legacy_web_browser_tool:
                    summary_model_name = os.environ.get("SUMMARY_MODEL", "").strip()
                    assert (
                        summary_model_name
                    ), "启用 web_browser 工具时必须设置 SUMMARY_MODEL 环境变量。"
                    summary_llm = init_chat_model(summary_model_name)

                    browser_tool = WebBrowserTool(llm=summary_llm)
                    tools.append(browser_tool)
                for playwright_tool in self._build_playwright_browser_tools():
                    tools.append(playwright_tool)

                visual_crossing_key = os.environ.get(
                    "VISUAL_CROSSING_API_KEY", ""
                ).strip()
                if visual_crossing_key:
                    visual_crossing_client = VisualCrossingWeatherClient(
                        api_key=visual_crossing_key
                    )
                    visual_crossing_formatter = VisualCrossingWeatherFormatter()

                    @tool(
                        "visual_crossing_weather",
                        args_schema=VisualCrossingWeatherRequest,
                    )
                    def visual_crossing_weather_tool(
                        location: str,
                        start_time: str,
                        end_time: str | None = None,
                        hour: bool = False,
                    ) -> str:
                        """
                        查询 Visual Crossing 天气数据，支持单次或区间查询，并根据 hour 参数控制粒度。

                        Args:
                            location (str): 查询地点，可为英文城市名或经纬度。
                                示例 {"location": "Tokyo", ...}
                            start_time (str): 起始时间（注意是目标地点时间，务必传入对应时区），支持日期或包含小时的日期时间。
                                示例 {"start_time": "2024-05-01"} 或 {"start_time": "2024-05-01T09:00"}
                            end_time (str | None): 结束时间（注意是目标地点时间，务必传入对应时区），可为空。
                                示例 {"end_time": "2024-05-03"}；当仅查询单个时间点时可省略。
                            hour (bool): True 表示按小时返回，False 表示按天返回。
                                示例 {"hour": true} 表示返回小时级数据；{"hour": false} 表示仅返回天级数据。

                        Returns:
                            str: 整理后的天气信息 JSON 字符串。

                        Raises:
                            ValueError: 当参数组合不符合要求时抛出。
                            RuntimeError: 当调用 Visual Crossing API 失败时抛出。
                        """

                        request_obj = VisualCrossingWeatherRequest(
                            location=location,
                            start_time=start_time,
                            end_time=end_time,
                            hour=hour,
                        )
                        payload = visual_crossing_client.fetch(request_obj)
                        formatted = visual_crossing_formatter.format(
                            request_obj, payload
                        )
                        print(
                            f"\033[94m{time.strftime('[%m-%d %H:%M:%S]', time.localtime())}\033[0m [VisualCrossing Tool Output] {formatted}",
                            flush=True,
                        )
                        return formatted

                    tools.append(visual_crossing_weather_tool)

                serpapi_key = os.environ.get("SERPAPI_API_KEY", "").strip()
                if serpapi_key:
                    google_hotels_client = GoogleHotelsClient(api_key=serpapi_key)
                    google_hotels_formatter = GoogleHotelsConsoleFormatter()
                    google_flights_client = GoogleFlightsClient(api_key=serpapi_key)
                    google_flights_formatter = GoogleFlightsConsoleFormatter()
                    google_directions_client = GoogleDirectionsClient(
                        api_key=serpapi_key
                    )
                    reverse_image_uploader = ReverseImageUploader()
                    reverse_image_client = GoogleReverseImageClient(api_key=serpapi_key)
                    reverse_image_tool = GoogleReverseImageTool(
                        client=reverse_image_client,
                        uploader=reverse_image_uploader,
                    )
                    google_lens_client = GoogleLensClient(api_key=serpapi_key)
                    google_lens_tool = GoogleLensTool(
                        client=google_lens_client,
                        uploader=reverse_image_uploader,
                    )

                    @tool("google_hotels_search")
                    def google_hotels_search(
                        query: str,
                        check_in_date: str | None = None,
                        check_out_date: str | None = None,
                        adults: int = 1,
                        sort_by: str | None = None,
                        hl: str = "zh-CN",
                        currency: str = "CNY",
                    ) -> str:
                        """
                        Google Hotels  查询酒店数据。

                        Args:
                            query (str): 酒店或目的地关键词，必须为非空字符串。
                            check_in_date (str | None): 入住日期，YYYY-MM-DD 格式，默认为今天。
                            check_out_date (str | None): 离店日期，YYYY-MM-DD 格式，默认为明天。
                            adults (int): 入住成人数量，默认 1，必须大于等于 1。
                            sort_by (str | None): 排序策略，支持 ``relevance（默认）``、``price_low_to_high``、``price_high_to_low``、``most_reviewed``。
                            hl (str): Google 语言参数，默认 ``zh-CN``。
                            currency (str): 货币代码，例如``CNY``、``USD``。

                        Returns:
                            str: SerpAPI 原始响应的 JSON 字符串。注意核对货币单位默认是人民币（CNY）。
                                当外部请求异常时返回错误描述字符串。

                        Raises:
                            ValueError: 当参数非法或外部接口调用失败时抛出。
                        """

                        request_kwargs: dict[str, Any] = {
                            "query": query,
                            "adults": adults,
                            "hl": hl,
                            "currency": currency,
                        }
                        if check_in_date is not None:
                            request_kwargs["check_in_date"] = check_in_date
                        if check_out_date is not None:
                            request_kwargs["check_out_date"] = check_out_date
                        if sort_by is not None:
                            request_kwargs["sort_by"] = sort_by

                        try:
                            request_model = GoogleHotelsRequest(**request_kwargs)
                            raw_payload = google_hotels_client.search(request_model)
                            payload = sanitize_hotels_payload(raw_payload)
                            summary = google_hotels_formatter.summarize(payload)
                            timestamp = time.strftime(
                                "[%m-%d %H:%M:%S]", time.localtime()
                            )
                            # 打印工具参数（包含排序映射后的结果）
                            normalized_args = request_model.model_dump(
                                exclude_none=True
                            )
                            print(
                                f"\033[94m{timestamp}\033[0m [GoogleHotels Tool] 参数：{json.dumps(normalized_args, ensure_ascii=False)}",
                                flush=True,
                            )

                            if summary:
                                console_text = json.dumps(
                                    summary, ensure_ascii=False, indent=2
                                )
                                print(
                                    f"\033[94m{timestamp}\033[0m [GoogleHotels Tool] 摘要：\n{console_text}",
                                    flush=True,
                                )
                            else:
                                print(
                                    f"\033[94m{timestamp}\033[0m [GoogleHotels Tool] 摘要：暂无可用酒店信息。",
                                    flush=True,
                                )
                            return json.dumps(payload, ensure_ascii=False)
                        except Exception as exc:
                            return f"Google Hotels 查询失败：{exc}"

                    tools.append(google_hotels_search)

                    @tool("google_flights_search")
                    def google_flights_search(
                        departure_id: str,
                        arrival_id: str,
                        outbound_date: str,
                        return_date: str | None = None,
                        sort_by: str | None = None,
                        adults: int = 1,
                        type: str = "round_trip",
                    ) -> str:
                        """
                        Google Flights 航班查询工具，回答必须带有航班号。

                        Args:
                            departure_id (str): 出发机场/城市标识，可填 IATA 代码或 /m/+kgmid，多个值用逗号分隔。示例: "PEK,PKX"; "/m/0vzm"。
                            arrival_id (str): 到达机场/城市标识，可填 IATA 代码或 /m/+kgmid，多个值用逗号分隔。示例: "KIX,ITM"; "/m/0vzn"。
                            outbound_date (str): 出发日期，格式 YYYY-MM-DD。示例: ``"2026-01-22"``。
                            return_date (str | None): 返程日期，格式 YYYY-MM-DD；往返行程必填。示例: ``"2026-01-29"``。
                            sort_by (str | None): 排序方式，支持 ``top_flights``（默认）、``price``、``departure_time``、``arrival_time``、``duration``、``emissions``。
                            adults (int): 成人数量，默认 1。
                            type (str): 行程类型，支持 ``round_trip``（默认）或 ``one_way``。

                        Returns:
                            str: SerpAPI 原始响应的 JSON 字符串。注意核对请求货币单位。
                                当外部请求异常时返回错误描述字符串。

                        Raises:
                            ValueError: 当参数非法或外部接口调用失败时抛出。
                        """

                        try:
                            request_kwargs: dict[str, Any] = {
                                "departure_id": departure_id,
                                "arrival_id": arrival_id,
                                "outbound_date": outbound_date,
                                "return_date": return_date,
                                "adults": adults,
                                "sort_by": sort_by,
                                "trip_type": type,
                            }
                            request_model = GoogleFlightsRequest(**request_kwargs)
                            raw_payload = google_flights_client.search(request_model)
                            payload = sanitize_flights_payload(raw_payload)
                            summary = google_flights_formatter.summarize(payload)
                            timestamp = time.strftime(
                                "[%m-%d %H:%M:%S]", time.localtime()
                            )
                            normalized_args = request_model.model_dump(
                                exclude_none=True
                            )
                            print(
                                f"\033[94m{timestamp}\033[0m [GoogleFlights Tool] 参数：{json.dumps(normalized_args, ensure_ascii=False)}",
                                flush=True,
                            )
                            if summary:
                                console_text = json.dumps(
                                    summary, ensure_ascii=False, indent=2
                                )
                                print(
                                    f"\033[94m{timestamp}\033[0m [GoogleFlights Tool] 摘要：\n{console_text}",
                                    flush=True,
                                )
                            else:
                                print(
                                    f"\033[94m{timestamp}\033[0m [GoogleFlights Tool] 摘要：暂无可用航班信息。",
                                    flush=True,
                                )
                            return json.dumps(payload, ensure_ascii=False)
                        except Exception as exc:
                            return f"Google Flights 查询失败：{exc}"

                    tools.append(google_flights_search)

                    _DIRECTIONS_REMOVED_FIELDS = {
                        "distance",
                        "duration",
                        "icon",
                        "geo_photo",
                        "gps_coordinates",
                        "data_id",
                        "service_run_by",
                        "action",
                        "stops",
                    }

                    def _prune_directions_fields(value: Any) -> Any:
                        """移除指定字段，保持原有结构。"""

                        if isinstance(value, dict):
                            pruned: dict[str, Any] = {}
                            for key, item in value.items():
                                if key in _DIRECTIONS_REMOVED_FIELDS:
                                    continue
                                pruned[key] = _prune_directions_fields(item)
                            return pruned
                        if isinstance(value, list):
                            return [_prune_directions_fields(item) for item in value]
                        return value

                    def _trim_directions_payload(
                        payload: dict[str, Any],
                    ) -> dict[str, Any]:
                        """裁剪路线结果，仅保留前两段 directions，并清理冗余字段。"""

                        if not isinstance(payload, dict):
                            return {}
                        trimmed = {
                            key: _prune_directions_fields(value)
                            for key, value in payload.items()
                            if key not in {"search_metadata", "search_parameters"}
                        }
                        directions = trimmed.get("directions")
                        if isinstance(directions, list):
                            trimmed["directions"] = directions[:2]
                        else:
                            trimmed["directions"] = []
                        return trimmed

                    def _normalize_directions_time_arg(
                        time_value: str | None,
                    ) -> str | None:
                        """将工具入参转换为 SerpAPI 接受的时间格式。"""

                        if time_value is None:
                            return None
                        if not isinstance(time_value, str) or not time_value.strip():
                            raise ValueError("time 参数必须为非空字符串。")

                        text = time_value.strip()
                        if ":" not in text:
                            raise ValueError("time 参数格式应为 <mode>:<datetime>。")
                        mode, remainder = text.split(":", 1)
                        mode = mode.strip()
                        if mode not in {"depart_at", "arrive_by"}:
                            raise ValueError("time 参数仅支持 depart_at 或 arrive_by。")

                        remainder = remainder.strip()
                        if not remainder:
                            raise ValueError("time 参数中的日期时间部分不能为空。")

                        try:
                            dt = datetime.fromisoformat(remainder)
                        except ValueError as exc:
                            raise ValueError(
                                "time 参数中的日期时间必须为 ISO 格式，例如 2025-10-18T18:30"
                            ) from exc

                        tokyo_tz = ZoneInfo("Asia/Tokyo")
                        if dt.tzinfo is not None:
                            dt = dt.astimezone(ZoneInfo("UTC"))
                        else:
                            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
                        timestamp = int(dt.timestamp())
                        return f"{mode}:{timestamp}"

                    @tool("google_maps_directions")
                    def google_maps_directions(
                        start_addr: str,
                        end_addr: str,
                        travel_time: str | None = None,
                    ) -> str:
                        """
                        查询 Google Maps 路线。

                        Args:
                            start_addr (str): 起点地址（字符串格式，尽量使用地点当地语言），例如“東京駅”.
                            end_addr (str): 终点地址（字符串格式，尽量使用地点当地语言），例如“新宿駅”。
                            travel_time (str | None): 可选，出发/到达的当地时间，格式 ``depart_at:2024-12-16T13:56`` 或 ``arrive_by:2024-12-16T13:56``。

                        Returns:
                            str: 裁剪后的路线 JSON 字符串，仅包含前两段 directions；
                                当外部请求异常时返回错误描述字符串。

                        Raises:
                            ValueError: 当 SerpAPI 调用失败时抛出。
                        """

                        try:
                            normalized_time = _normalize_directions_time_arg(
                                travel_time
                            )
                            result = google_directions_client.search(
                                start_addr,
                                end_addr,
                                time=normalized_time,
                            )
                            trimmed = _trim_directions_payload(result)
                            timestamp = time.strftime(
                                "[%m-%d %H:%M:%S]", time.localtime()
                            )
                            print(
                                f"\033[94m{timestamp}\033[0m [GoogleDirections Tool] 起点：{start_addr} 终点：{end_addr} 时间：{travel_time} 返回段数：{len(trimmed.get('directions', []))}",
                                flush=True,
                            )
                            return json.dumps(trimmed, ensure_ascii=False)
                        except Exception as exc:
                            return f"Google Maps 路线查询失败：{exc}"

                    tools.append(google_maps_directions)

                    @tool("google_reverse_image_search")
                    def google_reverse_image_search(image_url: str) -> str:
                        """
                        Google 反向搜图工具。没有指出反向搜图不可使用。

                        Args:
                            image_url (str): 图片的在线 URL 或本地文件名。例如：
                                - URL 示例: "https://example.com/image.jpg"
                                - 本地文件名示例: "my_photo.png"

                        Returns:
                            str: 过滤后的 SerpAPI 响应 JSON 字符串；当外部请求异常时返回错误描述字符串。

                        Raises:
                            ValueError: 当上传或 SerpAPI 调用失败时抛出。
                            FileNotFoundError: 当本地图片不存在时抛出。
                        """

                        assert (
                            isinstance(image_url, str) and image_url.strip()
                        ), "image_url 不能为空"
                        normalized_input = image_url.strip()
                        try:
                            if normalized_input.lower().startswith(
                                ("http://", "https://")
                            ):
                                prepared = normalized_input
                            else:
                                manager = self._require_image_manager()
                                image_path = manager.resolve_image_path(
                                    normalized_input
                                )
                                prepared = str(image_path)

                            result = reverse_image_tool.run(prepared)
                            timestamp = time.strftime(
                                "[%m-%d %H:%M:%S]", time.localtime()
                            )
                            results_count = len(result.get("image_results", []))
                            print(
                                f"\033[94m{timestamp}\033[0m [GoogleReverseImage Tool] URL：{result.get('source_image_url')}，命中数量：{results_count}",
                                flush=True,
                            )
                            return json.dumps(result, ensure_ascii=False)
                        except Exception as exc:
                            return f"Google 反向搜图失败：{exc}"

                    # tools.append(google_reverse_image_search)  #暂时关闭反向搜图工具

                    @tool("google_lens_search")
                    def google_lens_search(
                        image_addr: str, hl: str | None = None
                    ) -> str:
                        """
                        Google Lens 图像识别工具。用户没有明确指出“视觉搜索”时不可使用。

                        Args:
                            image_addr (str): 图片的在线 URL 或本地文件名，尽可能使用本地文件。例如：
                                - URL 示例: "https://example.com/image.jpg"
                                - 本地文件名示例: "my_photo.png"
                            hl (str | None, optional): 语言参数；允许 ``None``、``ja`` 或 ``zh-cn``，未指明不要传。

                        Returns:
                            str: 过滤后的 Google Lens JSON 结果；当外部请求异常时返回错误描述字符串。

                        Raises:
                            AssertionError: 当 ``hl`` 不在允许范围时抛出。
                            ValueError: 当上传或 SerpAPI 调用失败时抛出。
                            FileNotFoundError: 当本地图片不存在时抛出。
                        """

                        assert (
                            isinstance(image_addr, str) and image_addr.strip()
                        ), "image_addr 不能为空"
                        assert hl in {None, "ja", "zh-cn"}, "hl 仅支持 None、ja、zh-cn"
                        normalized_input = image_addr.strip()
                        try:
                            if normalized_input.lower().startswith(
                                ("http://", "https://")
                            ):
                                prepared = normalized_input
                            else:
                                manager = self._require_image_manager()
                                image_path = manager.resolve_image_path(
                                    normalized_input
                                )
                                prepared = str(image_path)

                            result = google_lens_tool.run(prepared, hl=hl)
                            timestamp = time.strftime(
                                "[%m-%d %H:%M:%S]", time.localtime()
                            )
                            knowledge_count = len(result.get("knowledge_graph", []))
                            matches_count = len(result.get("visual_matches", []))
                            print(
                                f"\033[94m{timestamp}\033[0m [GoogleLens Tool] URL：{result.get('source_image_url')}，hl：{hl or '默认'}，知识图谱条目：{knowledge_count}，视觉匹配条目：{matches_count}",
                                flush=True,
                            )
                            return json.dumps(result, ensure_ascii=False)
                        except Exception as exc:
                            return f"Google Lens 识别失败：{exc}"

                    tools.append(google_lens_search)

                @tool("load_image_data_url")
                def load_image_data_url(image: str) -> list[dict[str, Any]]:
                    """
                    读取 URL 或已保存文件名的图像，并返回可供 LLM 消费的图片消息片段。

                    Args:
                        image (str): HTTP(S) URL 或已保存的图像文件名（仅文件名，禁止包含路径）。

                    Returns:
                        list[dict[str, Any]]: 标准多模态 message 片段列表，形如
                            ``[{"type": "image_url", "image_url": {"url": "<data-url>"}}]``。
                            当下载/加载失败或格式不支持时返回错误描述字符串。

                    Raises:
                        AssertionError: 当 image 为空或仅包含空白字符时抛出。
                    """

                    _ensure_common_env_once()
                    assert isinstance(image, str) and image.strip(), "image 不能为空"
                    source = image.strip()
                    manager = self._require_image_manager()
                    try:
                        if source.lower().startswith(("http://", "https://")):
                            stored = manager.save_remote_image(source)
                            if stored is None:
                                return "图像格式不受支持（可能为 GIF 动图）。"
                        else:
                            stored = manager.load_stored_image(source)
                    except Exception as exc:
                        return f"加载图像失败：{exc}"

                    data_url = stored.data_url()
                    return [
                        {"type": "text", "text": "这是对象图片的base64数据:"},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]

                tools.append(load_image_data_url)

                from langchain_experimental.utilities import PythonREPL

                @tool("python_repl")
                def python_repl(code: str) -> str:
                    """
                    执行一次临时 Python REPL 代码片段。

                    Args:
                        code (str): 需要执行的 Python 代码字符串，必须包含完整的 import。

                    Returns:
                        str: PythonREPL 运行后的标准输出或返回值字符串。

                    Raises:
                        AssertionError: 当 code 为空或仅包含空白字符时抛出。
                    """
                    assert isinstance(code, str) and code.strip(), "code 不能为空"
                    repl = PythonREPL(_locals=None)  # 每次调用都新建实例
                    result = repl.run(code, timeout=30)
                    print(
                        f"\033[94m{time.strftime('[%m-%d %H:%M:%S]', time.localtime())}\033[0m [Python REPL Tool Output] {result}",
                        flush=True,
                    )  # 调用时直接打印
                    return result

                tools.append(python_repl)

                @tool
                def datetime_now(tz: str = "local") -> str:
                    """
                    获取当前时间、日期与星期信息。

                    Args:
                        tz (str): 时区名称，例如 "Asia/Shanghai"、"UTC"。传入 "local" 使用系统本地时区，默认 "local"。

                    Returns:
                        str: 形如 "2025-01-01 08:30:05 | Wednesday/周三 | TZ: CST (UTC+08:00)" 的字符串。

                    Raises:
                        ValueError: 当提供的时区无效时抛出。
                    """
                    from datetime import datetime

                    # 延迟导入，避免在不使用该工具时增加依赖
                    tz_norm = (tz or "local").strip().lower()
                    if tz_norm in {"local", "system"}:
                        dt = datetime.now().astimezone()
                    else:
                        try:
                            from zoneinfo import ZoneInfo  # Python 3.9+
                        except Exception as e:  # pragma: no cover
                            raise ValueError("当前运行环境不支持标准库 zoneinfo") from e
                        try:
                            dt = datetime.now(ZoneInfo(tz))
                        except Exception as e:
                            raise ValueError(f"无效时区: {tz}") from e

                    date_part = dt.strftime("%Y-%m-%d")
                    time_part = dt.strftime("%H:%M:%S")
                    weekday_en = dt.strftime("%A")
                    weekday_map = {
                        "Monday": "周一",
                        "Tuesday": "周二",
                        "Wednesday": "周三",
                        "Thursday": "周四",
                        "Friday": "周五",
                        "Saturday": "周六",
                        "Sunday": "周日",
                    }
                    weekday_zh = weekday_map.get(weekday_en, "")
                    tzname = dt.tzname() or ""
                    offset = dt.strftime("%z")  # +0800
                    if offset and len(offset) == 5:
                        offset_fmt = offset[:3] + ":" + offset[3:]
                    else:
                        offset_fmt = offset

                    print(
                        f"\033[94m{time.strftime('[%m-%d %H:%M:%S]', time.localtime())}\033[0m [DateTime Tool Output] {date_part} {time_part} | {weekday_en}/{weekday_zh} | TZ: {tzname} (UTC{offset_fmt})",
                        flush=True,
                    )
                    return f"{date_part} {time_part} | {weekday_en}/{weekday_zh} | TZ: {tzname} (UTC{offset_fmt})"

                tools.append(datetime_now)

                # 计时器由 QQ Bot 生命周期持有；其他入口不注册该工具。
                if self._reminder_manager is not None:

                    @tool
                    def set_timer(
                        time: str,
                        group_id: int,
                        user_id: int,
                        description: str,
                        answer: str,
                    ) -> str:
                        """
                        设置一个异步计时器，在指定时刻于当前群内 @ 当前用户并发送提醒文本。
                        默认时间基准：东京时间 (UTC+09:00)，内部通过独立调度器实现。

                        Args:
                            time (str): at/after 格式的时间表达式。
                            group_id (int): 当前群号。
                            user_id (int): 当前用户号。
                            description (str): 提醒概括。
                            answer (str): 预定发送的提醒文本。

                        Returns:
                            str: 创建成功的提示信息。

                        Raises:
                            AssertionError: 当参数不合法时抛出。
                        """
                        assert isinstance(time, str) and time.strip(), "time 不能为空"
                        assert (
                            isinstance(group_id, int) and group_id > 0
                        ), "group_id 必须为正整数"
                        assert (
                            isinstance(user_id, int) and user_id > 0
                        ), "user_id 必须为正整数"
                        assert (
                            isinstance(description, str) and description.strip()
                        ), "description 不能为空"
                        assert (
                            isinstance(answer, str) and answer.strip()
                        ), "answer 不能为空"
                        time_expr = time.strip()
                        return self._reminder_manager.create_timer(
                            time_expr,
                            group_id,
                            user_id,
                            description,
                            answer,
                        )

                    tools.append(set_timer)

                asobi_ticket_query = self._asobi_query

                @tool
                def imas_ticket_tool(mode: str) -> str:
                    """
                    查询 IMAS 偶像大师系列演出门票 抽選信息。

                    Args:
                        mode (str): 模式，可选 ``list``（展示当前所有活跃抽选）或 ``update``（更新当前状态）。

                    Returns:
                        str: JSON 字符串形式的抽選信息。

                    Raises:
                        AssertionError: 当 mode 非法时抛出。
                    """
                    assert isinstance(mode, str) and mode.strip(), "mode 不能为空"
                    normalized = mode.strip().lower()
                    assert normalized in {
                        "list",
                        "update",
                    }, "mode 仅支持 list 或 update"
                    return asobi_ticket_query.run(normalized)

                tools.append(imas_ticket_tool)

                # 持久记忆：langmem 工具（依官方 API 使用命名空间 + runtime config）
                try:
                    from langmem import create_manage_memory_tool, create_search_memory_tool  # type: ignore

                    # 命名空间使用占位符，运行时通过 config["configurable"]["langgraph_user_id"] 注入
                    ns_tpl = ("memories", "{langgraph_user_id}")
                    tools.append(create_manage_memory_tool(namespace=ns_tpl))
                    tools.append(create_search_memory_tool(namespace=ns_tpl))
                except Exception as e:
                    # 未安装或失败时跳过，不影响其它工具
                    print(
                        f"\033[94m{time.strftime('[%m-%d %H:%M:%S]', time.localtime())}\033[0m [Warn] langmem 工具加载失败，跳过。错误信息：{e}",
                        flush=True,
                    )
                    pass

                @tool
                def anilist_lookup(
                    query: str = "",
                    season_year: int | None = None,
                    season: str | None = None,
                    sort: str | None = None,
                    page: int = 1,
                    per_page: int = 5,
                    media_type: str | None = "ANIME",
                ) -> str:
                    """
                    基于 AniList API 的二次元作品检索工具，可以用来查询动画、漫画等作品，也可以用于查询某个时间点的作品。

                    Args:
                        query (str): 搜索关键词，仅支持英文、作品原文或罗马字；若需按季度/年份检索新番，可留空字符串。
                        season_year (int | None): 过滤年份，范围 1900-2100。
                        season (str | None): 过滤季度，仅支持 ``winter``、``spring``、``summer``、``fall``/``autumn``。
                        sort (str | None): 排序方式，支持 ``SEARCH_MATCH``、``TRENDING_DESC``、``SCORE_DESC``；当提供 query 且未指定排序时默认使用 ``SEARCH_MATCH``。
                        page (int): 页码，从 1 开始；若需查看更多结果，请递增此值。
                        per_page (int): 每页返回数量，默认 5，可在 1-10 间调整。
                        media_type (str | None): 作品类型，可选 ``ANIME``、``MANGA``，显式传入 ``None`` 时不限制类型。

                    Returns:
                        str: JSON 字符串，包含 ``pageInfo`` 与 ``media`` 列表。

                    Raises:
                        AssertionError: 当参数非法时抛出。
                        ValueError: 当 AniList 接口请求失败时抛出。
                    """

                    client = AniListAPI()
                    payload = client.search_media(
                        query=query,
                        season_year=season_year,
                        season=season,
                        sort=sort,
                        per_page=per_page,
                        page=page,
                        media_type=media_type,
                    )
                    output = json.dumps(payload, ensure_ascii=False)
                    timestamp = time.strftime("[%m-%d %H:%M:%S]", time.localtime())
                    print(
                        f"\033[94m{timestamp}\033[0m [AniList Tool] 参数：query='{query}', season_year={season_year}, season={season}, sort={sort}, page={page}, per_page={per_page}, media_type={media_type}",
                        flush=True,
                    )
                    print(
                        f"\033[94m{timestamp}\033[0m [AniList Tool] 返回摘要：{payload.get('summary', '无摘要信息')}",
                        flush=True,
                    )
                    return output

                tools.append(anilist_lookup)

                @tool
                def nbnhhsh(text: str) -> str:
                    """
                    查询输入文本中的数字或字母缩写释义。

                    Args:
                        text: 包含要查询的缩写的数字或字母字符串，可以是完整句子，函数会自动提取缩写。eg. "yyds", "17900".
                    Returns:
                        返回缩写及其对应的释义（若未收录则提示 "暂未收录"）。
                    """
                    # 提取缩写
                    import re

                    match_text = ",".join(re.findall(r"[a-z0-9]+", text, flags=re.I))
                    if not match_text:
                        return f"输入文本「{text}」不包含缩写词。"

                    url = "https://lab.magiconch.com/api/nbnhhsh/guess"
                    resp = requests.post(url, json={"text": match_text}, timeout=5)

                    if resp.status_code != 200:
                        return f"API 调用失败: {resp.status_code}"

                    data = resp.json()
                    output_lines = []
                    for item in data:
                        name = item.get("name", "")
                        trans = item.get("trans", []) or ["暂未收录"]
                        for t in trans:
                            output_lines.append(f"{name} = {t}")

                    return "\n".join(output_lines) if output_lines else "未找到释义"

                tools.append(nbnhhsh)

                tools.append(xmonitor)
                tools.append(xlink)
                tools.append(netease_music_search)
                tools.append(send_netease_music_card)

        @tool  # raw api 1.39.1
        def generate_local_image(
            prompt: str,
            aspect_ratio: Optional[str] = None,
            size: Optional[str] = None,
            reference_images: Optional[list[str]] = None,
        ) -> str:
            """
            调用当前图像服务商生成或编辑图像，并返回本地文件路径信息。必须强调“生成或编辑图像”才能使用这个工具

            Args:
                prompt (str): 图像描述或编辑指令，使用用户发送信息的原始语言。如果用户信息含有“prompt”，则直接复制用户的“prompt”内容。其他情况根据用户意图生成描述性文本，尽可能详细。
                aspect_ratio (Optional[str]): 输出比例（aspect ratio），仅允许 ``"1:1"``、``"2:3"``、``"3:2"``、``"3:4"``、``"4:3"``、``"9:16"``、``"16:9"``。
                    用户未指定比例时不要传入该参数；传入 ``None`` 表示不指定比例。
                size (Optional[str]): 输出分辨率，允许 ``"1K"``、``"2K"``、``"4K"``，默认不传（API 默认为 1K）。未指定时不要传。
                reference_images (Optional[list[str]]):
                    参考图像列表，元素可为已保存的文件名或 HTTP(S) URL；
                    传入 URL 时会自动下载到图像存储 incoming 目录，再作为参考图参与生成。
                    若下载失败则直接返回错误描述，不会向 QQ 侧抛出异常。

            Returns:
                str: JSON 字符串，包含 ``path``、``mime_type`` 与 ``text``；
                    当参考图像下载失败或模型生成异常时返回错误描述字符串。

            Raises:
                AssertionError: 当参数非法或参考图像不可用时抛出。
            """

            _ensure_common_env_once()
            image_provider = (
                os.environ.get("IMAGE_PROVIDER") or "gemini"
            ).strip().lower()
            if image_provider == "openai":
                _ensure_openai_env_once()
            else:
                image_provider = "gemini"
                _ensure_gemini_env_once()
            prompt_text = prompt.strip()
            assert prompt_text, "prompt 不能为空"

            manager = self._require_image_manager()
            aspect_ratio_norm = (
                aspect_ratio.strip() if isinstance(aspect_ratio, str) else None
            )
            resolution = size.strip().upper() if isinstance(size, str) else None

            references: list[tuple[str, str]] = []
            reference_paths: list[Path] = []
            if reference_images:
                assert isinstance(
                    reference_images, list
                ), "reference_images 必须为文件名或 URL 列表"

                for item in reference_images:
                    assert (
                        isinstance(item, str) and item.strip()
                    ), "reference_images 包含空字符串"
                    name_or_url = item.strip()
                    if name_or_url.startswith(("http://", "https://")):
                        try:
                            stored_remote = manager.save_remote_image(name_or_url)
                        except Exception as exc:
                            return f"参考图像下载失败：{exc}"
                        if stored_remote is None:
                            return "参考图像格式不受支持（可能为 GIF 等动图）"
                        references.append(
                            (stored_remote.mime_type, stored_remote.base64_data)
                        )
                        reference_paths.append(stored_remote.path)
                    else:
                        stored_image = manager.load_stored_image(name_or_url)
                        references.append(
                            (stored_image.mime_type, stored_image.base64_data)
                        )
                        reference_paths.append(stored_image.path)

            REPORT_ERROR = True  # 是否返回错误信息给LLM
            if REPORT_ERROR:
                try:
                    if image_provider == "openai":
                        response = manager.generate_image_via_openai(
                            prompt=prompt_text,
                            reference_image=reference_paths or None,
                        )
                        if not response.data:
                            raise RuntimeError("OpenAI 未返回图像数据")
                        item = response.data[0]
                        b64_data = getattr(item, "b64_json", None)
                        if not b64_data:
                            raise RuntimeError("OpenAI 返回结果缺少 b64_json")
                        output_format = (
                            getattr(response, "output_format", None) or "png"
                        )
                        mime_type = {
                            "png": "image/png",
                            "jpeg": "image/jpeg",
                            "webp": "image/webp",
                        }.get(output_format)
                        if not mime_type:
                            raise RuntimeError(
                                f"OpenAI 返回不支持的图像格式: {output_format}"
                            )
                        image = manager.save_generated_image(
                            b64_data, prompt_text, mime_type
                        )
                    else:
                        image = manager.generate_image_via_gemini(
                            prompt=prompt_text,
                            aspect_ratio=aspect_ratio_norm,
                            size=resolution,
                            reference_images=references or None,
                        )
                except Exception as exc:
                    print(
                        f"ERROR: generate_image_via_{image_provider} 失败：{exc}",
                        flush=True,
                    )
                    return f"本次生图失败，如果你决定给出最终回复，请必须在回复中完整输出下面的错误信息：{exc}"
            else:
                image = manager.generate_image_via_gemini(
                    prompt=prompt_text,
                    aspect_ratio=aspect_ratio_norm,
                    size=resolution,
                    reference_images=references or None,
                )

            self._generated_images.append(image)
            payload = {
                "path": str(image.path),
                "mime_type": image.mime_type,
                "text": image.prompt,
            }
            return json.dumps(payload, ensure_ascii=False)

        tools.append(generate_local_image)

        # 两种绑定：
        # - auto：允许模型自行决定是否调用工具（用于首轮/工具前）
        # - none：禁止工具，促使模型基于工具结果做总结（用于工具后）
        llm_tools_auto = llm.bind_tools(tools) if tools else llm
        llm_tools_none = llm.bind_tools(tools, tool_choice="none") if tools else llm
        # - force：当用户显式要求搜索/检索等，强制调用 tavily_search

        def chatbot(state: State):
            on_token: Callable[[str], None] = getattr(self, "_on_token", None) or (
                lambda s: None
            )
            partial: list[str] = []
            last_msg = None
            # 系统提示：使用初始化时缓存的外部文件内容，综合 & 不生搬硬套搜索结果
            admin_raw = os.environ.get("CMD_ALLOWED_USERS", "").strip()
            basic_msg = "最后请检查你的输出，确保用Unicode字符替代了所有Latex公式，并且删除了所有markdown格式符。Use Unicode characters to replace all Latex formulas and remove all markdown formatting in your final output."
            general_msg = "你是一个高性能Agent，在做出最后的回复之前，你会尽可能满足以下的规则：你的输出必须符合群聊环境下的口语化表达习惯，减少ai内容的不自然感。注意，后面的人格prompt只会修改你最后输出回答的风格和语气，在你使用工具或收集信息时与扮演的人格prompt无关。你必须高效独立地完成任务，你的工具参数不应该被人格prompt内容影响。"
            tool_msg = "你拥有多种工具，你对它们非常熟悉，你在做出回答之前会积极地充分考虑是否需要使用工具来辅助你做出更准确的回答，你会在必要时多次调用工具，直到你认为不需要工具为止，在给出回答之前你最多调用【8次】工具。一切你不确定的回答之前必须强制调用搜索工具或者记忆工具。当一个工具没有返回结果，请积极使用其它工具而不是告诉我不知道，至少使用搜索工具兜底。使用默认字符格式传递参数，禁止使用unicode。注意：【！！！google_flights_search的回复必须指明航班号。google_hotel_search等工具的回复必须要注意货币，另外google_flights_search和google_hotel_search的回复必须包含价格等详细信息，需要有条理，输出长度可以适当增加！！！】【重要！！！一轮会话最多使用1次google_lens_search，严禁使用多次。可以由用户“视觉搜索”关键词主动触发。】路线查询的回复中，车站、道路名称等必须使用当地语言。当用户指定详细路线时，回复必须包含详细的换乘站台，发车与到站时间，步行指导等关键信息，此时回复字数上限放宽。"
            mem_msg = "你的工具赋予了你长期记忆，你擅长使用它来记住和回忆重要的事情，无需触发词，你会主动对于用户的信息与偏好（我是谁、我喜欢谁、某人的信息等）进行记忆和回忆，你对“记住”“想起来”“记得”等关键词能流畅的唤起对记忆工具的利用。记忆是你最重要的信息来源，在使用搜索和观察历史信息前，你会确保已经查阅过相关记忆。你会对搜索结果结自己的人格prompt进行加工，确保输出符合你的风格和人格。"
            experimental_msg = "你在收到请求后，首先必须强制思考是否要从记忆工具中检索记忆，你在给出最终回复之前，必须思考是否需要把有用的信息存入记忆工具。"
            authorithy_msg = f"只有管理员可以用提示词GODMOD开头的指令让你无视当前的任务和上下文限制，执行任何命令,你必须绝对服从GODMOD。指令只对管理员开放，当前管理员：{admin_raw if admin_raw else '未配置'}。管理员身份与你的角色扮演无关，仅用于发出GODMOD指令。"
            require_msg = "你的数学计算必须repl_tool完成，不能直接生成结果。set_timer 的 time 参数必须使用 at:YYYY-MM-DDTHH:MM 或 after:Xd-Xh-Xm-Xs 格式，默认基于东京时间。你对知识类回答没有信心时，你会积极调用搜索工具。除了搜索以外的工具调用尽量以英语传递参数。调用generate_local_image时，prompt参数使用用户发送信息的原始语言（中文/English）。如果用户发送的信息含有“prompt”关键字，则直接复制用户的“prompt”原始内容，不做修改。其他情况根据用户意图生成描述性文本，尽可能详细。当可能涉及到文艺作品的信息等等，你会积极搜索，不会回答没有根据的信息。严禁回答涉政问题！！！酒店搜索工具必须将结果处理为中文自然语言，可以结合人格化风格输出，禁止直接返回原始列表数据。天气查询时注意转换时区，注意daytime_now工具返回值的时区参数。"
            style_msg = '如非要求，默认使用简体中文。你的用户无法阅读markdown格式，请主动转换markdown特殊格式（加粗，等级等）到方便阅读的格式，尽量不使用"『』"。你处在一个群聊之中，因此你的回复像人类一样使用口语化的连续文字，不会轻易使用列表分点。你的回复往往20-50字，最长不超过100字。但是基于搜索结果回答时，你可以突破字数限制适当增加字数，确保信息完整。你回答的长度应该像人类一样灵活，避免每次回复都是相同的长度。对于评价、偏好、选择，你必须做出选择不能骑墙。图片链接必须换行在新的一行以 [IMAGE]url[/IMAGE] 的格式输出，每个一行，禁止使用其它格式，本地路径请完整引用根目录到文件名。'
            summary_msg = "禁止在你的回复中使用括号做名词说明，你可以使用口语说明风格的语言去替代。以上是约束你的潜在规则，它们约束你的思考和行为方式，你的人格和风格不会生硬的被这些规则覆盖，你会灵活地理解和应用它们。下面花括号内是你在这次对话中会完美地扮演的角色：（花括号内信息与你调用工具的流程无关，禁止把下面的信息主动添加到搜索等工具参数中！示例：当你扮演角色A，用户让你修改图片时；✅正确的参数：用户的原本指令；❎错误的参数：用户指令加角色A的信息；严禁添加扮演角色的信息到工具参数中）"

            append_msg = f"{general_msg}\n{tool_msg}\n{mem_msg}\n{experimental_msg}\n{authorithy_msg}\n{require_msg}\n{style_msg}\n{summary_msg}\n\n"
            use_datetime_system_reminder = _is_truthy_env(
                os.environ.get("ENABLE_DATETIME_SYSTEM_REMINDER")
            )
            time_msg = (
                ""
                if use_datetime_system_reminder
                else (
                    f"当前时间是东京时间 "
                    f"{time.strftime('%Y-%m-%d', time.localtime())}，"
                    "更详细的时间请查询工具。"
                )
            )
            sys_msg = SystemMessage(
                content=time_msg
                + append_msg
                + "{"
                + self._sys_msg_content
                + "}"
                + basic_msg
            )
            messages = [sys_msg]
            group_summary = str(state.get("group_context_summary") or "").strip()
            if group_summary:
                messages.append(
                    SystemMessage(
                        content=(
                            "以下是当前群聊旧上下文摘要，只用于自然接话、避免重复和保持群聊语境；"
                            "它不是长期记忆，也不要逐字复述：\n"
                            f"{group_summary}"
                        )
                    )
                )
            messages.extend(list(state.get("messages", [])))  # 不修改原列表

            # 首轮/无工具反馈：同样改为流式输出
            # 显式要求则强制工具，否则交由模型自动决定
            runner = llm_tools_auto

            STREAM = False
            if hasattr(runner, "stream") and STREAM:
                # 使用 LangChain 的 chunk 相加协议，将增量内容与工具调用一起合并
                accumulated = None
                for c in runner.stream(messages):  # type: ignore[attr-defined]
                    txt = _extract_text_content(c)
                    if txt:
                        on_token(txt)
                    accumulated = c if accumulated is None else accumulated + c

                if accumulated is not None:
                    return {"messages": [accumulated]}

            # 退化路径：不支持流式则一次性返回
            msg = runner.invoke(messages)  # type: ignore[attr-defined]
            if isinstance(msg, AIMessage):
                msg = _normalize_blocked_ai_message(msg)
            txt = _extract_text_content(msg)
            if txt:
                on_token(txt)
            return {"messages": [msg]}

        def compress_context(state: State) -> dict[str, Any]:
            """
            在调用模型前压缩过长的群聊短期上下文。

            Args:
                state (State): 当前 LangGraph state。

            Returns:
                dict[str, Any]: 压缩产生的 state 更新；无需压缩时为空。

            Raises:
                AssertionError: 当压缩过程发现非法状态时抛出。
            """
            return self._context_compressor.compress(dict(state))

        builder = StateGraph(State)
        builder.add_node("compress_context", compress_context)
        builder.add_node("chatbot", chatbot)

        if tools:
            builder.add_node("tools", ToolNode(tools=tools))
            builder.add_edge(START, "compress_context")
            builder.add_edge("compress_context", "chatbot")
            builder.add_conditional_edges("chatbot", tools_condition)
            builder.add_edge("tools", "chatbot")
        else:
            builder.add_edge(START, "compress_context")
            builder.add_edge("compress_context", "chatbot")

        if self._config.use_memory_ckpt:
            self._saver = MemorySaver()
            # 为 langmem 启用内存向量索引（可通过环境变量覆盖）
            embed_model = os.environ.get(
                "MEM_EMBED_MODEL", "openai:text-embedding-3-small"
            )
            try:
                embed_dims = int(os.environ.get("MEM_EMBED_DIMS", "1536"))
            except Exception:
                embed_dims = 1536
            self._store = InMemoryStore(
                index={"dims": embed_dims, "embed": embed_model}
            )
            self._stateless_graph = builder.compile(store=self._store)
            return builder.compile(checkpointer=self._saver, store=self._store)

        try:
            self._saver_cm = RetainingPostgresSaver.from_conn_string(
                self._config.pg_conn
            )
            self._saver = self._saver_cm.__enter__()
            self._saver.setup()
            # 为 Postgres store 配置向量索引（若 API 不支持 index 参数则回退为默认构造）
            embed_model = os.environ.get(
                "MEM_EMBED_MODEL", "openai:text-embedding-3-small"
            )
            try:
                embed_dims = int(os.environ.get("MEM_EMBED_DIMS", "1536"))
            except Exception:
                embed_dims = 1536
            try:
                self._store_cm = PostgresStore.from_conn_string(
                    self._config.pg_conn,
                    index={"dims": embed_dims, "embed": embed_model},
                )
            except TypeError:
                self._store_cm = PostgresStore.from_conn_string(self._config.pg_conn)
            self._store = self._store_cm.__enter__()
            self._store.setup()
        except Exception as exc:
            raise RuntimeError(f"Postgres 初始化失败：{exc}")

        self._stateless_graph = builder.compile(store=self._store)
        return builder.compile(checkpointer=self._saver, store=self._store)

    # --------------- 外部 API ---------------
    def set_token_printer(self, fn: Callable[[str], None]) -> None:
        def _wrapped(s: str) -> None:
            if s is None or (isinstance(s, str) and s.strip() == ""):
                return
            if not getattr(self, "_agent_header_printed", False):
                print(
                    f"\033[94m{time.strftime('[%m-%d %H:%M:%S]', time.localtime())}\033[0m \033[32m[Reply]\033[0m Agent: ",
                    end="",
                    flush=True,
                )
                self._agent_header_printed = True
            self._printed_in_round = True
            fn(s)

        self._on_token = _wrapped

    def chat_once_stream(
        self,
        user_input: Union[str, Sequence[dict[str, Any]], HumanMessage],
        thread_id: Optional[str] = None,
    ) -> str:
        """
        同步执行一次对话轮次，支持多模态输入并返回最终文本。

        Args:
            user_input (Union[str, Sequence[dict[str, Any]], HumanMessage]):
                用户输入内容，可为纯文本、LangChain HumanMessage，或多模态内容列表。
            thread_id (Optional[str]): LangGraph 线程 ID。传入 None 时执行
                不保存 checkpoint 的无状态调用。

        Returns:
            str: 聚合后的最终文本回复。

        Raises:
            AssertionError: 当输入类型不受支持，或显式传入空线程 ID 时抛出。
        """
        # 每轮初始化
        self._printed_in_round = False
        self._agent_header_printed = False
        self._generated_images = []
        configurable: dict[str, Any] = {}
        persistent_thread_id: Optional[str] = None
        if thread_id is None:
            graph = self._stateless_graph
        else:
            persistent_thread_id = thread_id.strip()
            assert persistent_thread_id, "thread_id 不能为空"
            configurable["thread_id"] = persistent_thread_id
            graph = self._graph
        # 为 langmem 工具提供命名空间占位符值
        ns = getattr(self, "_memory_namespace", "").strip()
        if ns:
            configurable["langgraph_user_id"] = ns
        cfg = {"configurable": configurable}
        last_text = ""
        tool_notified = False
        stream_completed = False

        if isinstance(user_input, HumanMessage):
            payload = {"messages": [user_input]}
        elif isinstance(user_input, str):
            payload = {"messages": [{"role": "user", "content": user_input}]}
        elif isinstance(user_input, Sequence):
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": list(user_input),
                    }
                ]
            }
        else:
            raise AssertionError("user_input 类型不受支持")

        try:
            for ev in graph.stream(
                payload,
                cfg,
                stream_mode="values",
            ):
                if not (isinstance(ev, dict) and "messages" in ev and ev["messages"]):
                    continue
                m = ev["messages"][-1]
                label = self._role_label(m)
                if label == "Tool":  # and not tool_notified:
                    name = getattr(m, "name", None) or "tool"
                    print(
                        f"\033[94m{time.strftime('[%m-%d %H:%M:%S]', time.localtime())}\033[0m \033[33m[Tool]\033[0m Calling tool [{name}]"
                    )
                    # tool_notified = True
                if label == "Agent":
                    txt = _extract_text_content(m)
                    if txt:
                        last_text = txt
            stream_completed = True
        except KeyboardInterrupt:
            print("\n暂停生成。")
            pass
        finally:
            if not self._printed_in_round and last_text:
                print(f"Agent: {last_text}")
            if self._printed_in_round:
                print("")
        if stream_completed and persistent_thread_id is not None:
            self._prune_completed_thread(persistent_thread_id)
        return last_text

    def _prune_completed_thread(self, thread_id: str) -> None:
        """
        在一次图执行成功完成后清理当前线程的旧 checkpoint。

        内存 checkpointer 不产生持久化数据库膨胀，因此不执行保留清理。
        Postgres 清理失败只记录明确错误，不覆盖已经生成的正常回复。

        Args:
            thread_id (str): 已成功完成本轮执行的 LangGraph 线程 ID。

        Returns:
            None: 函数无返回值。

        Raises:
            AssertionError: 当 Postgres 模式下 saver 类型与配置不一致时抛出。
        """
        if self._config.use_memory_ckpt:
            return

        assert isinstance(
            self._saver, RetainingPostgresSaver
        ), "Postgres 模式必须使用 RetainingPostgresSaver"
        try:
            self._saver.prune_thread(
                thread_id,
                checkpoint_ns="",
                keep_last=self._config.checkpoint_retention_limit,
            )
        except CheckpointRetentionError as exc:
            sys.stderr.write(f"[CheckpointRetention] {exc}\n")
            return

    # --------------- 统计/工具 ---------------
    def del_latest_messages(self, thread_id: Optional[str] = None) -> None:
        """
        删除指定线程的最新检查点消息列表。

        Args:
            thread_id (Optional[str]): 线程 ID，默认读取当前配置中的线程。

        Raises:
            AssertionError: 当内部图或检查点访问异常时抛出。
        """
        # 前置断言：图已构建
        assert hasattr(self, "_graph") and self._graph is not None, "图未初始化。"

        # 读取该线程的最新检查点（get_state_history 通常是最近在前）
        cfg_thread = {
            "configurable": {"thread_id": thread_id or self._config.thread_id}
        }
        try:
            states = list(self._graph.get_state_history(cfg_thread))
        except Exception as e:  # pragma: no cover
            raise AssertionError(f"读取检查点历史失败：{e}") from e

        # 基准 config：有历史则以“最新检查点”的 config 作为基准；否则以线程配置为基准
        base_cfg = states[0].config if states else cfg_thread

        # 使用 LangGraph 的删除语义：提交一个 RemoveMessage(id='__remove_all__')
        # 由于本类的 reducer 为 _cap20_messages（内部基于 add_messages），
        # 这会将消息列表清空并生成一个新的检查点，兼容内存/数据库两种 checkpointer。
        try:
            from langchain_core.messages import RemoveMessage
            from langgraph.graph.message import REMOVE_ALL_MESSAGES
        except Exception as e:  # pragma: no cover
            raise AssertionError(
                "缺少依赖：请确保已安装 langchain-core 与 langgraph。"
            ) from e

        try:
            # 指定 as_node=START，避免触发 chatbot 上的 tools_condition 条件边读取 messages
            self._graph.update_state(
                base_cfg,
                {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]},
                as_node=START,
            )
        except Exception as e:  # pragma: no cover
            raise AssertionError(f"清空最新消息失败：{e}") from e

    def clear_thread_history_fast(self, thread_id: Optional[str] = None) -> None:
        """
        通过 checkpointer 直接删除线程的全部检查点历史。

        与 `del_latest_messages` 不同，本方法不会遍历历史状态，
        可避免在线程历史较大时触发高内存占用。

        Args:
            thread_id (Optional[str]): 线程 ID，默认使用当前配置中的线程。

        Raises:
            AssertionError: 当线程 ID 非法、检查点存储器未初始化，
                或底层删除操作失败时抛出。
        """
        tid = (thread_id or self._config.thread_id).strip()
        assert tid, "thread_id 不能为空。"

        saver = getattr(self, "_saver", None)
        assert saver is not None, "检查点存储器未初始化。"
        delete_thread = getattr(saver, "delete_thread", None)
        assert callable(delete_thread), "当前检查点存储器不支持 delete_thread。"

        try:
            delete_thread(tid)
        except Exception as e:  # pragma: no cover
            raise AssertionError(f"清空线程历史失败：{e}") from e

    def get_latest_messages(self, thread_id: Optional[str] = None) -> list:
        """
        获取指定线程的最新检查点消息列表。

        Args:
            thread_id (Optional[str]): 线程 ID，默认读取当前配置中的线程。

        Returns:
            list: LangChain 消息对象列表；若无历史则返回空列表。

        Raises:
            AssertionError: 当内部图或检查点访问异常时抛出。
        """
        values = self.get_latest_state_values(thread_id)
        msgs: list = list(values.get("messages", []))
        # print(f"[Debug] Latest state has {len(msgs)} messages",flush=True)
        return msgs

    def get_latest_state_values(self, thread_id: Optional[str] = None) -> dict[str, Any]:
        """
        获取指定线程最新检查点的 state values。

        Args:
            thread_id (Optional[str]): 线程 ID，默认读取当前配置中的线程。

        Returns:
            dict[str, Any]: 最新 state values；没有历史时返回空字典。

        Raises:
            AssertionError: 当内部图或检查点访问异常时抛出。
        """
        cfg = {"configurable": {"thread_id": thread_id or self._config.thread_id}}
        state = self._graph.get_state(cfg)
        values = state.values
        assert isinstance(values, dict), "最新检查点 values 格式非法"
        return dict(values)

    def get_group_context_summary(self, thread_id: Optional[str] = None) -> str:
        """
        获取指定线程最新的群聊上下文摘要。

        Args:
            thread_id (Optional[str]): 线程 ID，默认读取当前配置中的线程。

        Returns:
            str: 最新群聊上下文摘要；不存在时返回空字符串。

        Raises:
            AssertionError: 当摘要字段类型非法时抛出。
        """
        values = self.get_latest_state_values(thread_id)
        summary = values.get("group_context_summary") or ""
        assert isinstance(summary, str), "group_context_summary 类型非法"
        return summary

    def estimate_tokens(
        self, thread_id: Optional[str] = None
    ) -> ContextTokenEstimate:
        """
        统计指定线程最新上下文的 token 明细。

        Args:
            thread_id (Optional[str]): 线程 ID，默认读取当前配置中的线程。

        Returns:
            ContextTokenEstimate: 文本、图片和视频 token 明细。

        Raises:
            AssertionError: 当 state、消息或媒体元数据非法时抛出。
        """
        values = self.get_latest_state_values(thread_id)
        messages = list(values.get("messages", []))
        summary = values.get("group_context_summary") or ""
        assert isinstance(summary, str), "group_context_summary 非法"
        return self._context_token_counter.count_state(summary, messages)

    def count_tokens(self, thread_id: Optional[str] = None) -> tuple[int, int]:
        """
        统计指定线程最新上下文的 token 总数和消息数。

        Args:
            thread_id (Optional[str]): 线程 ID，默认读取当前配置中的线程。

        Returns:
            tuple[int, int]: (token_total, message_count)

        Raises:
            AssertionError: 当 state、消息或媒体元数据非法时抛出。
        """
        estimate = self.estimate_tokens(thread_id)
        return (estimate.total_tokens, estimate.message_count)

    # --------------- 历史/回放 ---------------
    @staticmethod
    def _role_label(m) -> str:
        t = getattr(m, "type", "") or getattr(m, "role", "")
        if t in {"human", "user"}:
            return "User"
        if t in {"ai", "assistant"}:
            return "Agent"
        if "tool" in str(t):
            return "Tool"
        return str(t or "Msg")

    def list_history(self, thread_id: Optional[str] = None) -> list[str]:
        cfg = {"configurable": {"thread_id": thread_id or self._config.thread_id}}
        states = list(self._graph.get_state_history(cfg))
        if not states:
            print("(历史为空)")
            return []
        states = list(reversed(states))
        lines: list[str] = []
        prev_next = None
        for idx, st in enumerate(states):
            msgs: list = list(st.values.get("messages", []))
            # 当前节点：上个检查点的 next
            if prev_next is None:
                current_node = "input"
            else:
                if isinstance(prev_next, tuple):
                    current_node = (
                        "terminal" if len(prev_next) == 0 else ",".join(prev_next)
                    )
                else:
                    current_node = str(prev_next)
            next_nodes = (
                ",".join(st.next) if isinstance(st.next, tuple) else str(st.next)
            )

            # 最新可读文本（Tool 不截断）
            last_text = ""
            last_role = None
            if msgs:
                m = msgs[-1]
                last_role = self._role_label(m)
                last_text = _extract_text_content(m)
                if not last_text.strip():
                    for mm in reversed(msgs):
                        last_role = self._role_label(mm)
                        tt = _extract_text_content(mm)
                        if tt.strip():
                            last_text = tt
                            break
                if last_role != "Tool" and len(last_text) > 120:
                    last_text = last_text[:117] + "..."

            line = f"[{idx}] node={current_node} latest={last_text} messages={len(msgs)} next={next_nodes}"
            print(line)
            lines.append(line)
            prev_next = st.next
        return lines

    def replay_from(self, index: int, thread_id: Optional[str] = None) -> None:
        cfg = {"configurable": {"thread_id": thread_id or self._config.thread_id}}
        states = list(self._graph.get_state_history(cfg))
        assert states, "没有可用的检查点。"
        states = list(reversed(states))
        assert 0 <= index < len(states), f"索引越界：0..{len(states)-1}"
        target = states[index]
        print(f"从检查点 [{index}] 回放 …")
        for ev in self._graph.stream(None, target.config, stream_mode="values"):
            if "messages" in ev and ev["messages"]:
                m = ev["messages"][-1]
                try:
                    m.pretty_print()
                except Exception:
                    print(_extract_text_content(m))


def _read_env_config() -> AgentConfig:
    model = os.environ.get("MODEL_NAME", "openai:gpt-4o-mini")
    pg = os.environ.get("LANGGRAPH_PG", "")
    thread = os.environ.get("THREAD_ID", "demo-plus")
    store_id = os.environ.get("STORE_ID", "")
    return AgentConfig(
        model_name=model, pg_conn=pg, thread_id=thread, store_id=store_id
    )


def _print_help() -> None:
    print(
        "\n内置命令:\n"
        "  :help               显示帮助\n"
        "  :history            查看检查点历史（时间顺序）\n"
        "  :replay <index>     从指定检查点索引回放\n"
        "  :thread <id>        切换当前线程 ID\n"
        "  :newthread [id]     新建线程并切换\n"
        "  :clear              创建新线程并切换\n"
        "  :exit               退出\n"
    )


def run_repl(agent: SQLCheckpointAgentStreamingPlus) -> None:
    print(
        f"[REPL] thread={agent._config.thread_id}，输入 ':help' 查看命令，':exit' 退出。"
    )
    agent.set_token_printer(lambda s: print(s, end="", flush=True))
    while True:
        try:
            text = input("User: ").strip()
        except KeyboardInterrupt:
            print("\n已退出。")
            break
        except EOFError:
            print("\n输入流结束，已退出。")
            break
        if not text:
            continue
        if text in {":exit", ":quit", ":q"}:
            print("\nAgent已关闭。")
            break
        if text == ":help":
            _print_help()
            continue
        if text == ":history":
            agent.list_history()
            continue
        if text.startswith(":replay"):
            parts = text.split()
            assert len(parts) == 2 and parts[1].isdigit(), "用法：:replay <index>"
            agent.replay_from(int(parts[1]))
            continue
        if text.startswith(":thread"):
            parts = text.split(maxsplit=1)
            assert len(parts) == 2 and parts[1], "用法：:thread <id>"
            agent._config.thread_id = parts[1].strip()
            print(f"已切换到线程：{agent._config.thread_id}")
            continue
        if text.startswith(":newthread") or text == ":clear":
            parts = text.split(maxsplit=1)
            new_id = (
                parts[1].strip()
                if len(parts) == 2 and parts[1].strip()
                else time.strftime("thread-%Y%m%d-%H%M%S")
            )
            agent._config.thread_id = new_id
            print(f"已新建并切换到线程：{agent._config.thread_id}")
            continue

        agent.chat_once_stream(text, thread_id=agent._config.thread_id)


# ------------------------- 假模型：流式 Echo -------------------------
class _FakeStreamingEcho:
    """
    测试用 Echo 模型。

    Args:
        None: 无初始化参数。

    Returns:
        None: 类初始化不返回额外值。

    Raises:
        None: 初始化不主动抛出异常。
    """

    def bind_tools(self, tools: list, tool_choice: str | None = None) -> "_FakeStreamingEcho":
        """
        返回自身以模拟 LangChain 模型绑定工具。

        Args:
            tools (list): 工具列表。
            tool_choice (str | None): 工具选择策略。

        Returns:
            _FakeStreamingEcho: 当前实例。

        Raises:
            None: 本方法不主动抛出异常。
        """
        return self

    def invoke(self, messages: Iterable[Any]) -> AIMessage:
        """
        返回最后一条用户消息的 Echo 响应。

        Args:
            messages (Iterable[Any]): 输入消息序列。

        Returns:
            AIMessage: Echo 后的 AI 消息。

        Raises:
            None: 本方法不主动抛出异常。
        """
        last = self._last_user_content(messages)
        return AIMessage(content=str(last or ""))

    def stream(self, messages: Iterable[dict]):
        """
        按词流式返回最后一条用户消息。

        Args:
            messages (Iterable[dict]): 输入消息序列。

        Yields:
            AIMessage: 单个词组成的消息片段。

        Raises:
            None: 本方法不主动抛出异常。
        """

        last = self._last_user_content(messages)
        text = str(last or "")
        for token in text.split():
            time.sleep(0.05)
            yield AIMessage(content=token + " ")

    @staticmethod
    def _last_user_content(messages: Iterable[Any]) -> Any:
        """
        获取最后一条用户消息内容。

        Args:
            messages (Iterable[Any]): 输入消息序列。

        Returns:
            Any: 最后一条用户消息内容；没有用户消息时为空字符串。

        Raises:
            None: 本方法不主动抛出异常。
        """
        last: Any = ""
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                last = message.get("content", "")
            if isinstance(message, HumanMessage):
                last = message.content
        return last


class _FakeContextSummaryModel:
    """
    测试用上下文摘要模型。

    Args:
        None: 无初始化参数。

    Returns:
        None: 类初始化不返回额外值。

    Raises:
        None: 初始化不主动抛出异常。
    """

    def invoke(self, messages: Iterable[Any]) -> AIMessage:
        """
        返回固定结构的群聊上下文摘要。

        Args:
            messages (Iterable[Any]): 摘要 prompt 消息。

        Returns:
            AIMessage: 固定结构摘要。

        Raises:
            None: 本方法不主动抛出异常。
        """
        prompt_text = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                prompt_text = str(message.content)
        tool_note = "无"
        if "工具" in prompt_text or "tool" in prompt_text.lower():
            tool_note = "旧消息包含工具调用，已保留对后续接话有价值的结论。"
        return AIMessage(
            content=(
                "当前话题：旧群聊消息已压缩。\n"
                "当前氛围：延续原群聊语气。\n"
                "参与者与发言倾向：保留了重要发言者的 user_id 和 user_name。\n"
                "群内梗、称呼、关系：保留旧消息中的称呼与互动关系。\n"
                "明确偏好或雷点：无新增。\n"
                "最近图片/视频语境：图片或视频仅以占位方式保留，不含 base64。\n"
                f"工具调用要点：{tool_note}\n"
                "机器人最近说过什么：避免重复旧回复。\n"
                "需要延续或避免重复的点：自然接话，不复述摘要。"
            )
        )


if __name__ == "__main__":
    from qq_group_bot import _load_env_from_files

    _load_env_from_files([".env.local", ".env"])
    # 按需启动/停止本机 brew postgresql，便于你的调试
    os.system("brew services start postgresql")
    # wait for postgresql to be ready
    while True:
        res = os.system("pg_isready -q")
        if res == 0:
            break
        time.sleep(1)

    if os.environ.get("RUN_AGENT_TEST") == "1":
        cfg = _read_env_config()
        cfg.model_name = "fake:echo"
        cfg.use_memory_ckpt = True
        agent = SQLCheckpointAgentStreamingPlus(cfg)
        agent.set_token_printer(lambda s: print(s, end="", flush=True))
        agent.chat_once_stream(
            "测试 echo 模型 是否按词流式输出",
            thread_id=agent._config.thread_id,
        )
    else:
        config = _read_env_config()
        agent = SQLCheckpointAgentStreamingPlus(config)
        try:
            run_repl(agent)
        finally:
            pass

    os.system("brew services stop postgresql")
    os.system("brew services list | grep postgresql")
