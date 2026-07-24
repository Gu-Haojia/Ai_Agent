"""
群聊上下文 token 估算模块。

统一统计消息文本、图片和视频的额定 token，供上下文压缩与 `/token`
命令复用。显示格式化与摘要 Prompt 构造不属于本模块职责。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

TextSanitizer = Callable[[str], str]


@dataclass(frozen=True)
class ContextTokenEstimate:
    """
    保存群聊上下文的本地 token 估算明细。

    Args:
        summary_text_tokens (int): 群聊摘要的文本 token 数。
        message_text_tokens (int): 当前消息列表的文本 token 数。
        image_tokens (int): 当前消息列表的图片额定 token 数。
        video_tokens (int): 当前消息列表的视频额定 token 数。
        image_count (int): 当前消息列表中的图片数量。
        video_count (int): 当前消息列表中的视频数量。
        video_seconds (int): 按每个视频向上取整后的累计计费秒数。
        message_count (int): 当前消息数量。

    Returns:
        None: dataclass 初始化不返回额外值。

    Raises:
        AssertionError: 当任一统计值为负数时抛出。
    """

    summary_text_tokens: int = 0
    message_text_tokens: int = 0
    image_tokens: int = 0
    video_tokens: int = 0
    image_count: int = 0
    video_count: int = 0
    video_seconds: int = 0
    message_count: int = 0

    def __post_init__(self) -> None:
        """
        校验 token 估算明细。

        Returns:
            None: 无返回值。

        Raises:
            AssertionError: 当任一统计值为负数时抛出。
        """
        values = (
            self.summary_text_tokens,
            self.message_text_tokens,
            self.image_tokens,
            self.video_tokens,
            self.image_count,
            self.video_count,
            self.video_seconds,
            self.message_count,
        )
        assert all(value >= 0 for value in values), "token 估算值不能为负数"

    @property
    def text_tokens(self) -> int:
        """
        返回摘要和消息正文的文本 token 总数。

        Returns:
            int: 文本 token 总数。

        Raises:
            None: 本属性不主动抛出异常。
        """
        return self.summary_text_tokens + self.message_text_tokens

    @property
    def total_tokens(self) -> int:
        """
        返回文本、图片和视频的 token 总数。

        Returns:
            int: 上下文 token 总数。

        Raises:
            None: 本属性不主动抛出异常。
        """
        return self.text_tokens + self.image_tokens + self.video_tokens


class ContextTokenCounter:
    """
    统一估算群聊文本与多模态上下文 token。

    Args:
        text_sanitizer (TextSanitizer): Base64 等大体积文本的清洗函数。
        token_encoder (Any | None): 可选的文本 token 编码器；为空时使用
            ``cl100k_base``。

    Returns:
        None: 初始化不返回额外值。

    Raises:
        AssertionError: 当清洗函数非法或文本编码器无法初始化时抛出。
    """

    IMAGE_TOKENS_PER_ITEM: int = 1120
    VIDEO_TOKENS_PER_SECOND: int = 102

    def __init__(
        self,
        text_sanitizer: TextSanitizer,
        token_encoder: Any | None = None,
    ) -> None:
        """
        初始化统一上下文 token 统计器。

        Args:
            text_sanitizer (TextSanitizer): Base64 等大体积文本的清洗函数。
            token_encoder (Any | None): 可选的文本 token 编码器。

        Returns:
            None: 无返回值。

        Raises:
            AssertionError: 当清洗函数非法或文本编码器无法初始化时抛出。
        """
        assert callable(text_sanitizer), "text_sanitizer 必须可调用"
        self._sanitize_text = text_sanitizer
        self._encoder = token_encoder or self._build_token_encoder()

    @staticmethod
    def _build_token_encoder() -> Any:
        """
        初始化文本 token 编码器。

        Returns:
            Any: ``cl100k_base`` 编码器。

        Raises:
            AssertionError: 当 tiktoken 缺失或编码器初始化失败时抛出。
        """
        try:
            import tiktoken  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise AssertionError("缺少依赖：请先安装 tiktoken。") from exc
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception as exc:  # pragma: no cover
            raise AssertionError("无法初始化 tiktoken 编码器 cl100k_base。") from exc

    def count_text_tokens(self, text: str) -> int:
        """
        统计清洗后原始文本的 token 数。

        Args:
            text (str): 待统计文本。

        Returns:
            int: 文本 token 数。

        Raises:
            AssertionError: 当文本类型非法或编码失败时抛出。
        """
        assert isinstance(text, str), "text 必须是字符串"
        sanitized = self._sanitize_text(text)
        try:
            return len(self._encoder.encode(sanitized))
        except Exception as exc:  # pragma: no cover
            raise AssertionError("上下文 token 编码失败。") from exc

    def truncate_text_tokens(self, text: str, max_tokens: int) -> str:
        """
        使用当前 token 编码器将文本截断到指定上限。

        Args:
            text (str): 需要截断的文本。
            max_tokens (int): 允许保留的最大 token 数。

        Returns:
            str: 截断后的文本；原文本未超限时返回清洗后的原文本。

        Raises:
            AssertionError: 当文本类型或 token 上限非法，或编码失败时抛出。
        """
        assert isinstance(text, str), "text 必须是字符串"
        assert isinstance(max_tokens, int) and max_tokens > 0, (
            "max_tokens 必须为正整数"
        )
        sanitized = self._sanitize_text(text)
        try:
            token_ids = self._encoder.encode(sanitized)
            if len(token_ids) <= max_tokens:
                return sanitized
            return self._encoder.decode(token_ids[:max_tokens]).rstrip()
        except Exception as exc:  # pragma: no cover
            raise AssertionError("上下文 token 截断失败。") from exc

    def count_state(
        self, summary: str, messages: Sequence[Any]
    ) -> ContextTokenEstimate:
        """
        统计摘要与消息组成的完整上下文。

        Args:
            summary (str): 当前群聊摘要。
            messages (Sequence[Any]): 当前 LangChain 消息序列。

        Returns:
            ContextTokenEstimate: 完整上下文 token 明细。

        Raises:
            AssertionError: 当摘要或消息序列类型非法时抛出。
        """
        assert isinstance(summary, str), "summary 必须是字符串"
        message_estimate = self.count_messages(messages)
        return ContextTokenEstimate(
            summary_text_tokens=self.count_text_tokens(summary),
            message_text_tokens=message_estimate.message_text_tokens,
            image_tokens=message_estimate.image_tokens,
            video_tokens=message_estimate.video_tokens,
            image_count=message_estimate.image_count,
            video_count=message_estimate.video_count,
            video_seconds=message_estimate.video_seconds,
            message_count=message_estimate.message_count,
        )

    def count_messages(self, messages: Sequence[Any]) -> ContextTokenEstimate:
        """
        统计消息列表的文本、图片和视频 token。

        Args:
            messages (Sequence[Any]): 当前 LangChain 消息序列。

        Returns:
            ContextTokenEstimate: 消息列表 token 明细。

        Raises:
            AssertionError: 当消息序列或视频时长非法时抛出。
        """
        assert isinstance(messages, Sequence), "messages 必须是序列"
        message_list = list(messages)
        text = "\n".join(self.message_to_text(message) for message in message_list)
        image_count = 0
        video_count = 0
        video_seconds = 0
        for message in message_list:
            content = getattr(message, "content", None)
            current_images, current_videos, current_seconds = self.count_content_media(
                content
            )
            image_count += current_images
            video_count += current_videos
            video_seconds += current_seconds
        return ContextTokenEstimate(
            message_text_tokens=self.count_text_tokens(text),
            image_tokens=image_count * self.IMAGE_TOKENS_PER_ITEM,
            video_tokens=video_seconds * self.VIDEO_TOKENS_PER_SECOND,
            image_count=image_count,
            video_count=video_count,
            video_seconds=video_seconds,
            message_count=len(message_list),
        )

    def message_to_text(self, message: Any) -> str:
        """
        将单条消息转换为 token 统计文本。

        Args:
            message (Any): LangChain 消息对象或兼容对象。

        Returns:
            str: 不含媒体原始数据和模型内部元数据的文本。

        Raises:
            None: 未知消息类型使用通用角色名称。
        """
        parts = [f"[{self.message_role(message)}]"]
        content_text = self.content_to_token_text(getattr(message, "content", None))
        if content_text:
            parts.append(content_text)
        if isinstance(message, AIMessage) and message.tool_calls:
            calls = self._sanitize_text(
                json.dumps(message.tool_calls, ensure_ascii=False, default=str)
            )
            parts.append(f"工具调用: {calls}")
        if isinstance(message, ToolMessage):
            tool_name = str(getattr(message, "name", "") or "").strip()
            if tool_name:
                parts.append(f"工具名: {tool_name}")
        return "\n".join(parts)

    @staticmethod
    def message_role(message: Any) -> str:
        """
        获取消息角色名称。

        Args:
            message (Any): LangChain 消息对象。

        Returns:
            str: 用于统计的中文角色名称。

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

    def content_to_token_text(self, content: Any) -> str:
        """
        将消息 content 转换为原始 token 统计文本。

        Args:
            content (Any): 消息 content 字段。

        Returns:
            str: 已排除媒体原始数据的文本。

        Raises:
            None: 未知结构使用清洗后的字符串表示。
        """
        if content is None:
            return ""
        if isinstance(content, str):
            return self._sanitize_text(content)
        if isinstance(content, (bytes, bytearray)):
            return f"[二进制内容已省略 len={len(content)}]"
        if isinstance(content, Sequence) and not isinstance(
            content, (str, bytes, bytearray)
        ):
            blocks = [self.block_to_token_text(block) for block in content]
            return "\n".join(item for item in blocks if item)
        return self._sanitize_text(str(content))

    def block_to_token_text(self, block: Any) -> str:
        """
        将单个内容块转换为原始 token 统计文本。

        Args:
            block (Any): 多模态内容块。

        Returns:
            str: 文本内容；媒体块返回空字符串。

        Raises:
            None: 未知结构使用清洗后的字符串表示。
        """
        if isinstance(block, dict):
            block_type = str(block.get("type") or "").lower()
            if block_type == "text":
                return self._sanitize_text(str(block.get("text") or ""))
            if block_type in {"image_url", "media", "video"}:
                return ""
            sanitized = dict(block)
            if "data" in sanitized:
                sanitized["data"] = "[BINARY_OMITTED]"
            return self._sanitize_text(
                json.dumps(sanitized, ensure_ascii=False, default=str)
            )
        block_type = str(getattr(block, "type", "") or "").lower()
        if block_type == "text":
            return self._sanitize_text(str(getattr(block, "text", "") or ""))
        if block_type == "image_url":
            return ""
        return self._sanitize_text(str(block))

    def count_content_media(self, content: Any) -> tuple[int, int, int]:
        """
        统计 content 中的图片数、视频数和视频计费秒数。

        Args:
            content (Any): LangChain 消息 content 字段。

        Returns:
            tuple[int, int, int]: 图片数、视频数、视频计费秒数。

        Raises:
            AssertionError: 当视频缺少合法时长或媒体类型不受支持时抛出。
        """
        if not isinstance(content, Sequence) or isinstance(
            content, (str, bytes, bytearray)
        ):
            return (0, 0, 0)
        image_count = 0
        video_count = 0
        video_seconds = 0
        for block in content:
            images, videos, seconds = self.count_block_media(block)
            image_count += images
            video_count += videos
            video_seconds += seconds
        return (image_count, video_count, video_seconds)

    @staticmethod
    def count_block_media(block: Any) -> tuple[int, int, int]:
        """
        统计单个内容块的媒体额定量。

        Args:
            block (Any): 多模态内容块。

        Returns:
            tuple[int, int, int]: 图片数、视频数、视频计费秒数。

        Raises:
            AssertionError: 当视频缺少合法时长或媒体类型不受支持时抛出。
        """
        if not isinstance(block, dict):
            block_type = str(getattr(block, "type", "") or "").lower()
            if block_type == "image_url":
                return (1, 0, 0)
            return (0, 0, 0)
        block_type = str(block.get("type") or "").lower()
        if block_type == "image_url":
            return (1, 0, 0)
        if block_type not in {"media", "video"}:
            return (0, 0, 0)
        mime_type = str(block.get("mime_type") or "").lower()
        if mime_type.startswith("image/"):
            return (1, 0, 0)
        assert block_type == "video" or mime_type.startswith(
            "video/"
        ), f"不支持的媒体 token 统计类型: {mime_type or 'unknown'}"
        duration = block.get("duration_seconds")
        assert isinstance(duration, (int, float)) and not isinstance(
            duration, bool
        ), "视频消息缺少 duration_seconds"
        assert math.isfinite(float(duration)) and float(duration) > 0, (
            "视频时长必须为正数"
        )
        return (0, 1, math.ceil(float(duration)))
