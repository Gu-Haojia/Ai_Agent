"""
网页浏览工具。

该模块提供一个用于抓取网页并结合大模型进行摘要或聚焦回答的 LangChain 工具。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
import time
from typing import Sequence
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, HttpUrl, field_validator


DEFAULT_HEADERS: dict[str, str] = {
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
    "Referer": "https://www.google.com/",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "cross-site",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": (
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) "
        "Gecko/20100101 Firefox/111.0"
    ),
}

_ALLOWED_CONTENT_TYPES: tuple[str, ...] = (
    "text/html",
    "application/json",
    "application/xml",
    "application/javascript",
    "text/plain",
)


@dataclass(frozen=True)
class LinkSnippet:
    """
    表示网页中的可用链接信息。

    Attributes:
        text (str): 链接文本内容。
        url (str): 绝对地址形式的链接。
    """

    text: str
    url: str


class WebBrowserInput(BaseModel):
    """
    WebBrowser 工具的结构化入参。

    Attributes:
        url (HttpUrl): 需要访问的网页 URL。
        question (str | None): 需要回答的具体问题；为空时输出摘要。
    """

    url: HttpUrl = Field(description="要访问的网页地址，必须包含协议。")
    question: str | None = Field(
        default=None,
        description="需要在网页中查找的具体问题；留空则返回摘要。",
    )

    @field_validator("question")
    @classmethod
    def _validate_question(cls, value: str | None) -> str | None:
        """
        清洗并校验 question 字段。

        Args:
            value (str | None): 原始问题文本。

        Returns:
            str | None: 去除首尾空白后的问题文本。

        Raises:
            ValueError: 当输入仅包含空白字符时抛出。
        """

        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("question 提供时必须包含非空白字符。")
        return stripped


class WebBrowserTool(BaseTool):
    """
    将网页抓取与 LangChain 模型结合的工具。

    支持摘要模式（未指定 question）与定向查找模式（指定 question）。
    """

    name: str = "web_browser"
    description: str = (
        "访问指定网页并在其内容中进行摘要或回答指定问题。"
        "使用该工具时需提供合法的网址，并可额外提供具体问题。"
    )
    args_schema: type[BaseModel] = WebBrowserInput

    def __init__(
        self,
        llm: BaseChatModel,
        embeddings: Embeddings | None = None,
        *,
        headers: dict[str, str] | None = None,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        timeout: float = 15.0,
    ) -> None:
        """
        初始化 WebBrowserTool。

        Args:
            llm (BaseChatModel): LangChain 聊天模型实例，用于生成最终回答。
            embeddings (Embeddings | None): 预留的向量化模型，目前未启用。
            headers (dict[str, str] | None): 自定义 HTTP 请求头。
            chunk_size (int): 文本分块大小，默认 2000。
            chunk_overlap (int): 文本分块之间的重叠字符数，默认 200。
            timeout (float): 请求超时秒数，默认 15 秒。

        Raises:
            AssertionError: 当 chunk_size 或 chunk_overlap 取值非法时抛出。
        """

        super().__init__()
        assert chunk_size > 0, "chunk_size 必须为正整数。"
        assert 0 <= chunk_overlap < chunk_size, "chunk_overlap 必须小于 chunk_size。"
        assert timeout > 0, "timeout 必须大于 0。"
        self._llm = llm
        self._embeddings = embeddings
        self._headers = headers or DEFAULT_HEADERS.copy()
        self._timeout = timeout
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def _run(
        self,
        url: str,
        question: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """
        同步执行工具逻辑。

        Args:
            url (str): 目标网页地址。
            question (str | None): 需要查找的问题内容；空值表示输出摘要。
            run_manager (CallbackManagerForToolRun | None): LangChain 回调管理器。

        Returns:
            str: LLM 基于网页内容生成的回答。

        Raises:
            AssertionError: 当 URL 不包含协议时抛出。
            RuntimeError: 当网络请求失败或响应内容类型不受支持时抛出。
        """

        normalized_url = str(url)
        assert normalized_url.startswith(
            ("http://", "https://")
        ), "URL 必须以 http 或 https 开头。"

        html = self._fetch_html(normalized_url)
        text, links = self._extract_text_and_links(
            html, normalized_url, question is None
        )
        question_text = None if question is None else str(question)
        if not text:
            raise RuntimeError("网页正文内容为空，无法生成结果。")

        documents = [
            Document(page_content=chunk, metadata={"index": index})
            for index, chunk in enumerate(self._splitter.split_text(text))
        ]
        if not documents:
            raise RuntimeError("网页内容切分后为空，无法生成结果。")

        context = self._select_context(documents, question_text)
        prompt = self._build_prompt(normalized_url, question_text, context, links)
        chain = prompt | self._llm | StrOutputParser()
        result = chain.invoke({}, run_manager=run_manager)
        timestamp = time.strftime("[%m-%d %H:%M:%S]", time.localtime())
        print(f"{timestamp} [WebBrowserTool] {result}")
        return result

    async def _arun(  # pragma: no cover - 同步工具未实现异步接口
        self,
        url: str,
        question: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """
        异步接口占位，以保持与 BaseTool API 一致。

        Args:
            url (str): 目标网页地址。
            question (str | None): 需要回答的问题。
            run_manager (CallbackManagerForToolRun | None): LangChain 回调管理器。

        Returns:
            str: 该方法未实现，将始终抛出异常。

        Raises:
            NotImplementedError: 始终抛出，提示该工具暂不支持异步调用。
        """

        raise NotImplementedError("WebBrowserTool 暂不支持异步调用。")

    def _fetch_html(self, url: str) -> str:
        """
        获取网页 HTML 文本。

        Args:
            url (str): 目标地址。

        Returns:
            str: HTML 原文。

        Raises:
            RuntimeError: 当请求失败或返回状态码非 2xx 时抛出。
        """

        try:
            response = requests.get(url, headers=self._headers, timeout=self._timeout)
        except requests.RequestException as exc:
            raise RuntimeError(f"请求网页失败: {exc}") from exc

        if not 200 <= response.status_code < 300:
            raise RuntimeError(f"网页返回异常状态码: {response.status_code}")

        content_type = (response.headers.get("Content-Type") or "").split(";", 1)[0]
        if content_type and content_type not in _ALLOWED_CONTENT_TYPES:
            raise RuntimeError(f"网页 Content-Type 不受支持: {content_type}")

        response.encoding = response.encoding or "utf-8"
        return response.text

    def _extract_text_and_links(
        self,
        html: str,
        base_url: str,
        summary_mode: bool,
    ) -> tuple[str, Sequence[LinkSnippet]]:
        """
        从 HTML 中提取可读文本与链接。

        Args:
            html (str): 网页 HTML 文本。
            base_url (str): 用于解析相对链接的基准地址。
            summary_mode (bool): 是否处于摘要模式；摘要时仅遍历 body。

        Returns:
            tuple[str, Sequence[LinkSnippet]]: 清洗后的正文文本与链接列表。
        """

        soup = BeautifulSoup(html, "html.parser")
        root: Tag | BeautifulSoup | None = soup.body if summary_mode else soup
        if root is None:
            return "", []

        texts: list[str] = []
        links: list[LinkSnippet] = []
        for element in root.descendants:
            if isinstance(element, NavigableString):
                parent = element.parent
                if not isinstance(parent, Tag):
                    continue
                if parent.name in {"script", "style", "noscript", "svg"}:
                    continue
                content = element.strip()
                if content:
                    texts.append(content)
                continue

            if not isinstance(element, Tag):
                continue
            if element.name == "a":
                anchor_text = element.get_text(separator=" ", strip=True)
                if not anchor_text:
                    continue
                href_raw = element.get("href")
                if not href_raw:
                    continue
                absolute_href = self._resolve_href(href_raw, base_url)
                if absolute_href is None:
                    continue
                img_alt = ""
                image = element.find("img", alt=True)
                if image is not None:
                    img_alt = image.get("alt", "").strip()
                combined_text = f"{anchor_text} {img_alt}".strip()
                if combined_text:
                    links.append(LinkSnippet(text=combined_text, url=absolute_href))
                    texts.append(f"[{combined_text}]({absolute_href})")

        # 去除重复链接，仅保留前 20 个用于提示
        unique_links = self._deduplicate_links(links, limit=20)
        normalized_text = self._normalize_whitespace(" ".join(texts))
        return normalized_text, unique_links

    def _resolve_href(self, href: str, base_url: str) -> str | None:
        """
        解析相对链接为绝对地址。

        Args:
            href (str): 原始链接。
            base_url (str): 基准地址。

        Returns:
            str | None: 解析后的绝对地址；失败时返回 None。
        """

        cleaned = href.strip()
        if not cleaned:
            return None
        parsed = urlparse(cleaned)
        if parsed.scheme in {"http", "https"}:
            return cleaned
        if cleaned.startswith("#"):
            return None
        try:
            joined = urljoin(base_url, cleaned)
        except ValueError:
            return None
        parsed_joined = urlparse(joined)
        if parsed_joined.scheme not in {"http", "https"}:
            return None
        return joined

    def _deduplicate_links(
        self,
        links: Sequence[LinkSnippet],
        limit: int,
    ) -> Sequence[LinkSnippet]:
        """
        根据链接地址去重并保留给定数量的链接。

        Args:
            links (Sequence[LinkSnippet]): 原始链接列表。
            limit (int): 最大保留数量。

        Returns:
            Sequence[LinkSnippet]: 去重后的链接列表。
        """

        seen: set[str] = set()
        deduped: list[LinkSnippet] = []
        for item in links:
            if item.url in seen:
                continue
            seen.add(item.url)
            deduped.append(item)
            if len(deduped) >= limit:
                break
        return deduped

    def _normalize_whitespace(self, text: str) -> str:
        """
        统一文本中的空白字符。

        Args:
            text (str): 原始文本。

        Returns:
            str: 仅保留单个空格的文本。
        """

        return re.sub(r"\s+", " ", text).strip()

    def _select_context(
        self,
        documents: Sequence[Document],
        question: str | None,
    ) -> str:
        """
        基于问题选择最相关的若干文本片段。

        Args:
            documents (Sequence[Document]): 网页分块后的文档列表。
            question (str | None): 需要回答的问题；为空时执行摘要模式。

        Returns:
            str: 拼接后的上下文文本。
        """

        if question is None:
            return "\n".join(doc.page_content for doc in documents[:4])

        scored = sorted(
            documents,
            key=lambda doc: self._score_chunk(doc.page_content, question),
            reverse=True,
        )
        top_chunks = [doc.page_content for doc in scored[:4] if doc.page_content]
        if not top_chunks:
            # 无匹配内容时退回至前几段，避免空上下文
            return "\n".join(doc.page_content for doc in documents[:4])
        return "\n".join(top_chunks)

    def _score_chunk(self, chunk: str, question: str) -> float:
        """
        对文本分块进行简单相关性打分。

        Args:
            chunk (str): 文本片段。
            question (str): 用户问题。

        Returns:
            float: 相关性得分，数值越大表示越相关。
        """

        if not chunk:
            return 0.0
        chunk_tokens = self._tokenize(chunk)
        question_tokens = self._tokenize(question)
        overlap = chunk_tokens & question_tokens
        if not overlap:
            return 0.0
        base_score = len(overlap)
        density = base_score / max(len(chunk_tokens), 1)
        return float(base_score + density)

    def _tokenize(self, text: str) -> set[str]:
        """
        将文本拆分为用于匹配的 token 集合。

        Args:
            text (str): 原始文本。

        Returns:
            set[str]: 去重后的 token 集合。
        """

        tokens = re.findall(r"[A-Za-z0-9\u4e00-\u9fa5]+", text.lower())
        return {token for token in tokens if len(token) > 1}

    def _build_prompt(
        self,
        url: str,
        question: str | None,
        context: str,
        links: Sequence[LinkSnippet],
    ):
        """
        构建 LangChain Prompt。

        Args:
            url (str): 目标网址。
            question (str | None): 用户问题。
            context (str): 网页上下文文本。
            links (Sequence[LinkSnippet]): 去重后的链接列表。

        Returns:
            ChatPromptTemplate: 组合好的提示模板。
        """

        links_markdown = "\n".join(
            f"- [{item.text}]({item.url})" for item in links[:5]
        )
        if not links_markdown:
            links_markdown = "- 无合适链接"

        task_text = question or "请对页面内容进行中文摘要。"
        system_prompt = (
            "你是一名中文网页阅读助手，需要根据提供的网页片段完成任务。"
            "请引用关键信息，保持事实准确，禁止凭空生成内容。"
        )
        human_prompt = (
            "目标网址: {url}\n"
            "任务说明: {task}\n\n"
            "相关片段:\n{context}\n\n"
            "候选链接:\n{links}\n\n"
            "请给出结构化回答，并在末尾保留标题为 'Relevant Links:' 的 Markdown 列表，"
            "该列表内容必须取自候选链接。"
        )

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "human",
                    human_prompt,
                ),
            ]
        ).partial(url=url, task=task_text, context=context, links=links_markdown)
