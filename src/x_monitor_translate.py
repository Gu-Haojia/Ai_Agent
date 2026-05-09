"""
XMonitor 推文文本翻译工具。

本模块使用 Google 官方 Gemini API 将推文正文翻译为简体中文，并在渲染前
根据环境变量选择不翻译、仅翻译或原文译文对照模式。
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional, Protocol, Sequence

from src.x_monitor_render import XRenderedTweet

TRANSLATION_MODE_ENV = "X_MONITOR_TRANSLATION_MODE"
TRANSLATION_MODEL = "gemini-3-flash-preview"


class XTweetTranslationMode:
    """
    定义 XMonitor 推文翻译模式。
    """

    NONE = "none"
    TRANSLATED = "translated"
    BILINGUAL = "bilingual"

    @classmethod
    def normalize(cls, value: Optional[str]) -> str:
        """
        标准化环境变量中的翻译模式。

        Args:
            value (Optional[str]): 环境变量原始值。

        Returns:
            str: 标准化后的翻译模式。

        Raises:
            AssertionError: 当模式不是受支持值时抛出。
        """
        raw = (value or "").strip().lower()
        if raw in {"", "0", "off", "none", "no", "false", "不翻译"}:
            return cls.NONE
        if raw in {"1", "translated", "translate", "only", "zh", "仅翻译"}:
            return cls.TRANSLATED
        if raw in {"2", "bilingual", "compare", "dual", "both", "对照"}:
            return cls.BILINGUAL
        raise AssertionError(
            f"{TRANSLATION_MODE_ENV} 仅支持 none/translated/bilingual"
        )

    @classmethod
    def from_env(cls) -> str:
        """
        从环境变量读取翻译模式。

        Returns:
            str: 标准化后的翻译模式。

        Raises:
            AssertionError: 当环境变量模式非法时抛出。
        """
        return cls.normalize(os.environ.get(TRANSLATION_MODE_ENV))


class TweetTextTranslator(Protocol):
    """
    推文文本翻译器协议。
    """

    def translate_texts(self, texts: Sequence[str]) -> list[str]:
        """
        批量翻译文本。

        Args:
            texts (Sequence[str]): 原文列表。

        Returns:
            list[str]: 与原文一一对应的简体中文译文。

        Raises:
            AssertionError: 当输入或输出不符合预期时抛出。
        """
        ...


class GeminiTweetTranslator:
    """
    使用 Google 官方 Gemini API 翻译推文文本。
    """

    def __init__(
        self, client: Optional[Any] = None, model: str = TRANSLATION_MODEL
    ) -> None:
        """
        初始化 Gemini 翻译器。

        Args:
            client (Optional[Any]): 可注入的 Google GenAI 客户端。
            model (str): Gemini 模型名称。

        Returns:
            None: 构造函数无返回值。

        Raises:
            AssertionError: 当模型名称为空时抛出。
        """
        assert model.strip(), "Gemini 翻译模型不能为空"
        self._client = client
        self._model = model.strip()

    def translate_texts(self, texts: Sequence[str]) -> list[str]:
        """
        将推文文本批量翻译为简体中文。

        Args:
            texts (Sequence[str]): 原文列表。

        Returns:
            list[str]: 与原文一一对应的简体中文译文。

        Raises:
            AssertionError: 当输入为空、API 响应格式异常或译文数量不一致时抛出。
        """
        originals = [text.strip() for text in texts]
        assert originals, "翻译文本不能为空"
        assert all(originals), "翻译文本不能包含空字符串"
        client = self._client or self._create_client()
        prompt = self._build_prompt(originals)
        response = client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=self._build_config(),
        )
        payload = json.loads(self._response_text(response))
        translations = payload.get("translations")
        assert isinstance(translations, list), "Gemini 翻译响应缺少 translations"
        assert len(translations) == len(originals), "Gemini 翻译数量与输入不一致"
        result = [str(item).strip() for item in translations]
        assert all(result), "Gemini 翻译结果不能包含空字符串"
        return result

    def _create_client(self) -> Any:
        """
        创建 Google GenAI 客户端。

        Returns:
            Any: Google GenAI 客户端。

        Raises:
            AssertionError: 当缺少可用 Gemini 凭据时抛出。
        """
        from google import genai

        api_key = (
            os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY")
            or ""
        ).strip()
        if api_key:
            return genai.Client(api_key=api_key)

        use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower()
        project = (os.environ.get("GOOGLE_CLOUD_PROJECT") or "").strip()
        location = (os.environ.get("GOOGLE_CLOUD_LOCATION") or "").strip()
        assert use_vertex in {"1", "true", "yes", "on"} and project and location, (
            "缺少 Gemini 可用环境变量，请设置 GOOGLE_API_KEY / GEMINI_API_KEY / "
            "GOOGLE_GENERATIVE_AI_API_KEY，或配置 Vertex AI 项目与区域。"
        )
        return genai.Client(vertexai=True, project=project, location=location)

    def _build_config(self) -> Any:
        """
        构造 Gemini 文本生成配置。

        Returns:
            Any: GenerateContentConfig 实例。

        Raises:
            None: 本方法不主动抛出异常。
        """
        from google.genai import types

        return types.GenerateContentConfig(
            temperature=0,
            responseMimeType="application/json",
        )

    def _build_prompt(self, texts: Sequence[str]) -> str:
        """
        构造翻译提示词。

        Args:
            texts (Sequence[str]): 原文列表。

        Returns:
            str: Gemini 提示词。

        Raises:
            None: 本方法不主动抛出异常。
        """
        payload = json.dumps(list(texts), ensure_ascii=False)
        return (
            "请将下面 JSON 数组中的 X/Twitter 推文正文翻译为简体中文。"
            "保留换行、URL、@用户名、#话题标签、emoji 和专有名词。"
            "只返回 JSON 对象，格式必须为 {\"translations\":[...]}，"
            "数组长度必须与输入完全一致。\n"
            f"输入：{payload}"
        )

    def _response_text(self, response: Any) -> str:
        """
        从 Gemini 响应中提取文本。

        Args:
            response (Any): Gemini API 响应对象。

        Returns:
            str: 响应文本。

        Raises:
            AssertionError: 当响应缺少文本内容时抛出。
        """
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        candidates = list(getattr(response, "candidates", None) or [])
        assert candidates, "Gemini 翻译响应缺少候选结果"
        content = getattr(candidates[0], "content", None)
        parts = list(getattr(content, "parts", None) or [])
        texts = [
            str(part_text).strip()
            for part in parts
            for part_text in [getattr(part, "text", None)]
            if isinstance(part_text, str) and part_text.strip()
        ]
        assert texts, "Gemini 翻译响应缺少文本内容"
        return "\n".join(texts)


class XRenderedTweetTextTranslator:
    """
    将翻译结果应用到可渲染推文树。
    """

    def __init__(self, translator: Optional[TweetTextTranslator] = None) -> None:
        """
        初始化渲染文本翻译器。

        Args:
            translator (Optional[TweetTextTranslator]): 可注入的文本翻译器。

        Returns:
            None: 构造函数无返回值。

        Raises:
            None: 本方法不主动抛出异常。
        """
        self._translator = translator or GeminiTweetTranslator()

    def apply(self, tweet: XRenderedTweet, mode: str) -> None:
        """
        按指定模式转换推文树中的正文。

        Args:
            tweet (XRenderedTweet): 需要渲染的推文树根节点。
            mode (str): 翻译模式。

        Returns:
            None: 直接修改推文树文本。

        Raises:
            AssertionError: 当模式非法或翻译结果数量不一致时抛出。
        """
        normalized = XTweetTranslationMode.normalize(mode)
        if normalized == XTweetTranslationMode.NONE:
            return
        targets = [item for item in self._collect_tweets(tweet) if item.text.strip()]
        if not targets:
            return
        originals = [item.text for item in targets]
        translations = self._translator.translate_texts(originals)
        assert len(translations) == len(targets), "翻译结果数量与推文数量不一致"
        for target, original, translation in zip(targets, originals, translations):
            target.text = self._compose_text(original, translation, normalized)

    def _collect_tweets(self, tweet: XRenderedTweet) -> list[XRenderedTweet]:
        """
        收集推文树中的唯一推文节点。

        Args:
            tweet (XRenderedTweet): 根推文。

        Returns:
            list[XRenderedTweet]: 按遍历顺序排列的唯一推文。

        Raises:
            None: 本方法不主动抛出异常。
        """
        ordered: list[XRenderedTweet] = []
        seen: set[str] = set()

        def visit(current: XRenderedTweet) -> None:
            """
            递归访问推文引用树。

            Args:
                current (XRenderedTweet): 当前推文节点。

            Returns:
                None: 无返回值。

            Raises:
                None: 本函数不主动抛出异常。
            """
            if current.tweet_id in seen:
                return
            seen.add(current.tweet_id)
            ordered.append(current)
            for reference in current.references:
                visit(reference.tweet)

        visit(tweet)
        return ordered

    def _compose_text(self, original: str, translation: str, mode: str) -> str:
        """
        根据翻译模式组合正文。

        Args:
            original (str): 原文。
            translation (str): 简体中文译文。
            mode (str): 翻译模式。

        Returns:
            str: 用于渲染的正文。

        Raises:
            AssertionError: 当模式非法时抛出。
        """
        normalized = XTweetTranslationMode.normalize(mode)
        if normalized == XTweetTranslationMode.TRANSLATED:
            return translation.strip()
        if normalized == XTweetTranslationMode.BILINGUAL:
            return f"{original.strip()}\n\n简中翻译：\n{translation.strip()}"
        raise AssertionError(f"不支持的翻译模式: {mode}")
