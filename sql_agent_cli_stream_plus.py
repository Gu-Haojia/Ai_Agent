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
import threading
import requests

from typing_extensions import TypedDict
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage
from image_storage import GeneratedImage, ImageStorageManager
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
from src.anilist_client import AniListAPI, ANILIST_MEDIA_SORTS
from src.timer_reminder import TimerReminderManager
from src.asobi_ticket_agent import AsobiTicketQuery

ANILIST_SORT_CHOICES_TEXT: str = ", ".join(ANILIST_MEDIA_SORTS)

# ---- 环境校验：仅在首次需要时检查，避免重复消耗 ----
_ENV_COMMON_CHECKED: bool = False
_ENV_OPENAI_CHECKED: bool = False
_ENV_GEMINI_CHECKED: bool = False


def _ensure_common_env_once() -> None:
    """
    进程级通用环境校验，仅首次调用时执行，确保激活 `.venv` 虚拟环境。

    Returns:
        None: 函数无返回值。

    Raises:
        AssertionError: 当未激活 `.venv` 虚拟环境时抛出。
    """
    global _ENV_COMMON_CHECKED
    if _ENV_COMMON_CHECKED:
        return
    assert os.environ.get("VIRTUAL_ENV") or sys.prefix.endswith(
        ".venv"
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


def _ensure_gemini_env_once() -> None:
    """
    Gemini 相关环境校验，仅首次需要 Gemini 时执行，兼容多种环境变量命名。

    Returns:
        None: 函数无返回值。

    Raises:
        AssertionError: 当缺少可用的 Gemini API Key 时抛出。
    """
    global _ENV_GEMINI_CHECKED
    if _ENV_GEMINI_CHECKED:
        return
    key = (
        os.environ.get("GEMINI_API_KEY")
    )
    assert key, "缺少 GOOGLE_API_KEY / GEMINI_API_KEY 环境变量。"
#    if not os.environ.get("GOOGLE_API_KEY"):
#        os.environ["GOOGLE_API_KEY"] = key
    _ENV_GEMINI_CHECKED = True


_BASE64_DATA_URL_PATTERN = re.compile(
    r"(data:[^;]+;base64,)[0-9A-Za-z+/=_-]+", re.IGNORECASE
)


def _sanitize_for_logging(payload: object) -> str:
    """
    将待写入日志的对象转换为字符串，并将 base64 段落替换为占位符。

    Args:
        payload (object): 待记录的消息对象，可以是字符串、列表或消息实例。

    Returns:
        str: 已将 base64 内容脱敏为 ``[BASE64...]`` 的字符串。

    Raises:
        None.
    """

    text = str(payload)

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
    return sanitized


def _apply_format(text: str) -> str:
    """
    根据环境变量 FORMAT 统一移除 Markdown 加粗符号。

    Args:
        text (str): 原始字符串。

    Returns:
        str: 已去除 ``**`` 的字符串。
    """

    # 强制开启格式化：默认写入 FORMAT=1，再执行去除加粗符号。
    if os.environ.get("FORMAT") != "1":
        os.environ["FORMAT"] = "1"
    return text.replace("**", "")


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
    if isinstance(content, Sequence) and not isinstance(
        content, (str, bytes, bytearray)
    ):
        text = _blocks_to_text(content)
        if text:
            return _apply_format(text)

    return None


def _extract_text_for_token_count(message: Any) -> str:
    """
    提取用于 token 统计的文本。

    优先复用 ``_extract_text_content``（遇到纯工具调用时会返回 None），
    当未取到文本时回退为格式化后的字符串化结果，避免 join 时出现 None。

    Args:
        message (Any): LangChain 消息对象。

    Returns:
        str: 可用于 token 统计的文本。
    """

    text = _extract_text_content(message)
    if text is not None and text != "":
        return text
    return _apply_format(str(message))



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
    """基于内置 `add_messages` 的长度控制合并器：仅保留最近 n 条。

    先使用 `add_messages(prev, new)` 完成标准的消息合并（与内置追加行为一致），
    再对结果做截断，返回最后 n 条，避免改变既有消息规范化与合并语义。
    同时会将新增消息内容追加到日志文件中，便于后续排查与回放。

    Args:
        prev (list|None): 既有消息列表。
        new (list|object): 新增消息（单条或列表）。

    Returns:
        list: 合并后保留最后 n 条的消息列表。
    """
    """
    #暂时不使用
    LENGTH_LIMIT = int(os.environ.get("MESSAGE_LENGTH_LIMIT", 20))
    if not isinstance(LENGTH_LIMIT, int) or LENGTH_LIMIT < 10:
        LENGTH_LIMIT = 20
    """
    LENGTH_LIMIT = 20
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

    if len(combined) < LENGTH_LIMIT:
        # print(f"\n\n[Debug] Merged messages: {combined}", flush=True)
        # print(f"\n\n[Debug] Length of combined messages: {len(combined)}", flush=True)
        return combined
    start_msg = combined[-LENGTH_LIMIT] if len(combined) >= LENGTH_LIMIT else None
    # print(isinstance(start_msg, ToolMessage), flush=True)
    if isinstance(start_msg, HumanMessage):
        # print(f"\n\n[Debug] Merged messages: {combined[-LENGTH_LIMIT:]}", flush=True)
        # print(f"\n\n[Debug] Length of combined messages: {len(combined[-LENGTH_LIMIT:])}", flush=True)
        return combined[-LENGTH_LIMIT:]
    # 向后找到下一条 HumanMessage
    for i in range(len(combined) - LENGTH_LIMIT - 1, -1, -1):
        if isinstance(combined[i], HumanMessage):
            # print(f"\n\n[Debug] Merged messages: {combined[i:]}", flush=True)
            # print(f"\n\n[Debug] Length of combined messages: {len(combined[i:])}", flush=True)
            return combined[i:]
    return []


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


class State(TypedDict):
    """Agent 的图状态。"""

    messages: Annotated[list, _cap_messages]


@dataclass
class AgentConfig:
    """Agent 运行配置。"""

    model_name: str = "openai:gpt-4o-mini"
    pg_conn: str = ""
    thread_id: str = "demo-plus"
    use_memory_ckpt: bool = False
    enable_tools: bool = False
    # 用于持久记忆（langmem）命名空间的 store 隔离标识，由环境变量 STORE_ID 注入
    store_id: str = ""


class _ReminderStore:
    """
    简单的提醒持久化存储（JSON 文件）。

    结构：列表，每个元素为字典：
        {"ts": int, "group_id": int, "user_id": int, "description": str}

    文件路径可通过环境变量 `REMINDER_STORE_FILE` 覆盖，默认 `.qq_reminders.json`。
    所有操作具备进程内线程安全（基于 `threading.Lock`）。
    """

    _LOCK = threading.Lock()

    def __init__(self, path: str) -> None:
        assert isinstance(path, str) and path.strip(), "持久化文件路径无效"
        self._path = os.path.abspath(path)

    def _read_all(self) -> list[dict]:
        """读取全部记录；不存在返回空列表，格式异常抛出断言。"""
        if not os.path.isfile(self._path):
            return []
        with open(self._path, "r", encoding="utf-8") as f:
            raw = f.read()
        if not raw.strip():
            return []
        data = json.loads(raw)
        assert isinstance(data, list), "提醒存储文件格式应为列表"
        return data

    def _write_all(self, items: list[dict]) -> None:
        """原子写入全部记录。"""
        tmp = self._path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self._path)

    @staticmethod
    def _validate(rec: dict) -> None:
        ts = rec.get("ts")
        gid = rec.get("group_id")
        uid = rec.get("user_id")
        desc = rec.get("description")
        ans = rec.get("answer")
        assert isinstance(ts, int) and ts > 0, "ts 必须为正整数时间戳"
        assert isinstance(gid, int) and gid > 0, "group_id 必须为正整数"
        assert isinstance(uid, int) and uid > 0, "user_id 必须为正整数"
        assert isinstance(desc, str) and desc.strip(), "description 不能为空"
        assert isinstance(ans, str) and ans.strip(), "answer 不能为空"

    def add(self, rec: dict) -> None:
        """追加一条提醒记录。"""
        self._validate(rec)
        with self._LOCK:
            items = self._read_all()
            items.append(
                {
                    "ts": int(rec["ts"]),
                    "group_id": int(rec["group_id"]),
                    "user_id": int(rec["user_id"]),
                    "description": str(rec["description"]),
                    "answer": str(rec["answer"]),
                }
            )
            self._write_all(items)

    def prune_and_get_active(self, now_ts: int) -> list[dict]:
        """清理过期项并返回未过期记录（ts > now_ts）。"""
        assert isinstance(now_ts, int) and now_ts >= 0
        with self._LOCK:
            skip_num = 0
            items = self._read_all()
            active: list[dict] = []
            for r in items:
                try:
                    self._validate(r)
                except AssertionError:
                    # 跳过非法项
                    continue
                if int(r["ts"]) > now_ts:
                    active.append(
                        {
                            "ts": int(r["ts"]),
                            "group_id": int(r["group_id"]),
                            "user_id": int(r["user_id"]),
                            "description": str(r["description"]),
                            "answer": str(r["answer"]),
                        }
                    )
                else:
                    skip_num += 1

            print(
                f"\033[94m{time.strftime('[%m-%d %H:%M:%S]', time.localtime())}\033[0m "
                f"\033[33m[TimerStore]\033[0m 已清理过期提醒 {skip_num} 条，剩余有效提醒 {len(active)} 条。",
                flush=True,
            )
            # 覆盖写入仅保留有效项
            self._write_all(active)
            return active

    def remove_one(
        self, ts: int, group_id: int, user_id: int, description: str, answer: str
    ) -> None:
        """移除第一条与参数完全匹配的记录（若不存在则忽略）。"""
        with self._LOCK:
            items = self._read_all()
            idx = -1
            for i, r in enumerate(items):
                try:
                    if (
                        int(r.get("ts")) == int(ts)
                        and int(r.get("group_id")) == int(group_id)
                        and int(r.get("user_id")) == int(user_id)
                        and str(r.get("description")) == str(description)
                        and str(r.get("answer")) == str(answer)
                    ):
                        idx = i
                        break
                except Exception:
                    continue
            if idx >= 0:
                items.pop(idx)
                self._write_all(items)


class SQLCheckpointAgentStreamingPlus:
    """多轮工具 + 强化综合 的流式 Agent。"""

    def __init__(self, config: AgentConfig) -> None:
        # 仅首次进行通用环境校验
        _ensure_common_env_once()

        dry_run = os.environ.get("DRY_RUN") == "1"
        self._config = config
        if dry_run:
            self._config.use_memory_ckpt = True

        if not self._config.use_memory_ckpt:
            assert self._config.pg_conn, "必须通过 LANGGRAPH_PG 提供 Postgres 连接串。"
        if self._config.model_name != "fake:echo":
            _ensure_model_env_once(self._config.model_name)

        env_tools = os.environ.get("ENABLE_TOOLS")
        if env_tools is None:
            self._enable_tools = config.enable_tools
        else:
            self._enable_tools = env_tools in {"1", "true", "True"}

        # 预先读取并缓存系统提示内容（从外部文件），避免每轮重复IO
        self._sys_msg_content: str = self._load_sys_msg_content()
        # 提醒存储：用于计时器持久化
        self._reminder_store = _ReminderStore(
            os.environ.get("REMINDER_STORE_FILE", ".qq_reminders.json")
        )
        self._reminder_scheduler = TimerReminderManager(self._reminder_store)
        self._asobi_query = AsobiTicketQuery()
        self._image_manager: Optional[ImageStorageManager] = None
        self._generated_images: list[GeneratedImage] = []

        self._graph = self._build_graph()
        self._printed_in_round: bool = False
        # 当前持久记忆命名空间（供 langmem 工具使用）；由外部在请求前设置
        self._memory_namespace: str = ""
        # Agent 启动时恢复并调度尚未过期的提醒
        self._reminder_scheduler.restore_pending()

    def shutdown(self) -> None:
        """
        停止 Agent 的后台调度资源。

        Raises:
            None.
        """
        if isinstance(self._reminder_scheduler, TimerReminderManager):
            self._reminder_scheduler.stop()

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
            print(
                f"\033[94m{time.strftime('[%m-%d %H:%M:%S]', time.localtime())}\033[0m [SysInfo] 已加载文件: {abs_path}，长度 {len(content)} 字符。"
            )
            # 打印头尾各50字符,仅输出文本不要格式符号
            print(
                f"\033[94m{time.strftime('[%m-%d %H:%M:%S]', time.localtime())}\033[0m [SysInfo] Prompt内容预览: {content[:50].replace(chr(10), ' ')} ... {content[-50:].replace(chr(10), ' ')}"
            )
        assert content and content.strip(), "系统提示文件内容为空。"
        return content

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
                from langchain_tavily import TavilySearch

                if os.environ.get("TAVILY_API_KEY"):
                    tools = [TavilySearch(max_results=3)]


                summary_model_name = os.environ.get("SUMMARY_MODEL", "").strip()
                assert (
                    summary_model_name
                ), "启用 web_browser 工具时必须设置 SUMMARY_MODEL 环境变量。"
                summary_llm = init_chat_model(summary_model_name)

                browser_tool = WebBrowserTool(llm=summary_llm)
                tools.append(browser_tool)

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
                            timestamp = time.strftime("[%m-%d %H:%M:%S]", time.localtime())
                            # 打印工具参数（包含排序映射后的结果）
                            normalized_args = request_model.model_dump(exclude_none=True)
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
                            timestamp = time.strftime("[%m-%d %H:%M:%S]", time.localtime())
                            normalized_args = request_model.model_dump(exclude_none=True)
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
                            normalized_time = _normalize_directions_time_arg(travel_time)
                            result = google_directions_client.search(
                                start_addr,
                                end_addr,
                                time=normalized_time,
                            )
                            trimmed = _trim_directions_payload(result)
                            timestamp = time.strftime("[%m-%d %H:%M:%S]", time.localtime())
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
                            if normalized_input.lower().startswith(("http://", "https://")):
                                prepared = normalized_input
                            else:
                                manager = self._require_image_manager()
                                image_path = manager.resolve_image_path(normalized_input)
                                prepared = str(image_path)

                            result = reverse_image_tool.run(prepared)
                            timestamp = time.strftime("[%m-%d %H:%M:%S]", time.localtime())
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
                    def google_lens_search(image_url: str, hl: str | None = None) -> str:
                        """
                        Google Lens 图像识别工具。用户没有明确指出“视觉搜索”时不可使用。

                        Args:
                            image_url (str): 图片的在线 URL 或本地文件名。例如：
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
                            isinstance(image_url, str) and image_url.strip()
                        ), "image_url 不能为空"
                        assert hl in {None, "ja", "zh-cn"}, "hl 仅支持 None、ja、zh-cn"
                        normalized_input = image_url.strip()
                        try:
                            if normalized_input.lower().startswith(("http://", "https://")):
                                prepared = normalized_input
                            else:
                                manager = self._require_image_manager()
                                image_path = manager.resolve_image_path(normalized_input)
                                prepared = str(image_path)

                            result = google_lens_tool.run(prepared, hl=hl)
                            timestamp = time.strftime("[%m-%d %H:%M:%S]", time.localtime())
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
                    return [{"type": "text", "text": "这是对象图片的base64数据:"},{"type": "image_url", "image_url": {"url": data_url}}]

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

                # 计时器：群内 @ 提醒（异步非阻塞）
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
                        time (str): 时间表达式，支持
                            - `at:YYYY-MM-DDTHH:MM`
                            - `after:Xd-Xh-Xm-Xs`（可缺省任意片段，但必须携带单位）。
                        group_id (int): 当前Group。
                        user_id (int): 当前User_id。
                        description (str): 提供给工具的简要的提醒概括。
                        answer (str): 提醒到时间后预定发送给用户的，符合当前说话风格的提醒内容。

                    Returns:
                        str: 是否创建成功的提示信息。

                    Raises:
                        AssertionError: 当参数不合法时抛出。
                    """
                    # 参数校验（显式断言，禁止模糊降级）
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
                    assert isinstance(answer, str) and answer.strip(), "answer 不能为空"

                    time_expr = time.strip()
                    return self._reminder_scheduler.create_timer(
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

        @tool #raw api 1.39.1
        def generate_local_image(
            prompt: str,
            aspect_ratio: Optional[str] = None,
            size: Optional[str] = None,
            reference_images: Optional[list[str]] = None,
        ) -> str:
            """
            调用 Gemini 接口生成或编辑图像，并返回本地文件路径信息。必须强调“生成或编辑图像”才能使用这个工具

            Args:
                prompt (str): 图像描述或编辑指令，必须包含清晰主体与风格，一定要在prompt中体现用户需求，越详细越好，如果用户指定了prompt，直接复制即可，例如 “Transform the photo into a high-end studio portrait in the style of Apple executive headshots.The subject is shown in a half-body composition, wearing professional yet minimalist attire, with a natural and confident expression.Use soft directional lighting to gently highlight the facial features, leaving subtle catchlights in the eyes.The background shouldbe a smooth gradient in neutral tones (light gray or off-white), with clear separation between subject and background.Add a touch of refined film grain for texture, and keep the atmosphere calm, timeless, and sophisticated.Composition should follow minimalist principles, with negative space and non-centered framing for a modern look.--no text, logos, distracting objects, clutter”
                aspect_ratio (Optional[str]): 输出比例（aspect ratio），仅允许 ``"1:1"``、``"2:3"``、``"3:2"``、``"3:4"``、``"4:3"``、``"9:16"``、``"16:9"``。
                    用户未指定比例时不要传入该参数；传入 ``None`` 表示不指定比例。
                size (Optional[str]): 输出分辨率，允许 ``"1K"``、``"2K"``、``"4K"``，默认不传（API 默认为 1K）。未指定时不要传。
                reference_images (Optional[list[str]]):
                    参考图像列表，元素可为已保存的文件名或 HTTP(S) URL；
                    传入 URL 时会自动下载到图像存储 incoming 目录，再作为参考图参与生成。
                    若下载失败则直接返回错误描述，不会向 QQ 侧抛出异常。

            Returns:
                str: JSON 字符串，包含 ``path``、``mime_type`` 与 ``text``；
                    当参考图像下载失败或格式不支持时返回错误描述字符串。

            Raises:
                AssertionError: 当参数非法或参考图像不可用时抛出。
                RuntimeError: 当 Gemini 未返回有效图像时抛出。
                ValueError: 当接口调用失败时抛出。
            """

            _ensure_common_env_once()
            _ensure_gemini_env_once()
            prompt_text = prompt.strip()
            assert prompt_text, "prompt 不能为空"

            manager = self._require_image_manager()
            aspect_ratio_norm = (
                aspect_ratio.strip() if isinstance(aspect_ratio, str) else None
            )
            resolution = size.strip().upper() if isinstance(size, str) else None

            references: list[tuple[str, str]] = []
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
                    else:
                        stored_image = manager.load_stored_image(name_or_url)
                        references.append(
                            (stored_image.mime_type, stored_image.base64_data)
                        )

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
            try:
                from langchain_core.messages import SystemMessage

                basic_msg = "Please output plain text only. No Markdown, no code fences, no special formatting. The output will be parsed as plain text. Do not use any Markdown symbols(*、#、**、```、$$、$). Do not output TeX math ($...$, $$...$$, \(...\), \[...\]). Convert all math expressions into Unicode math symbols (e.g., α, β, ≤, ≥, ×, ∞, √x, x², xₙ). If any Markdown or TeX appears, remove it. 删除输出内容的markdown和tex，公式转成unicode! 删除输出内容的markdown和tex，公式转成unicode! 删除输出内容的markdown和tex，公式转成unicode! 删除输出内容的markdown和tex，公式转成unicode! 删除输出内容的markdown和tex，公式转成unicode! "
                general_msg = "你是一个高性能Agent，在做出最后的回复之前，你会尽可能满足以下的规则：你的输出必须符合群聊环境下的口语化表达习惯，减少ai内容的不自然感。注意，后面的人格prompt只会修改你最后输出回答的风格和语气，在你使用工具或收集信息时与扮演的人格prompt无关。你必须高效独立地完成任务，你的工具参数不应该被人格prompt内容影响。"
                tool_msg = "你拥有多种工具，你对它们非常熟悉，你在做出回答之前会积极地充分考虑是否需要使用工具来辅助你做出更准确的回答，你会在必要时多次调用工具，直到你认为不需要工具为止。一切你不确定的回答之前必须强制调用搜索工具或者记忆工具。当一个工具没有返回结果，请积极使用其它工具而不是告诉我不知道，至少使用搜索工具兜底。使用默认字符格式传递参数，禁止使用unicode。注意：【！！！google_flights_search的回复必须指明航班号。google_hotel_search等工具的回复必须要注意货币，另外google_flights_search和google_hotel_search的回复必须包含价格等详细信息，需要有条理，输出长度可以适当增加！！！】【重要！！！由于使用次数限制，只有在用户明确提到“视觉搜索”，才可以使用google_lens_search，除此以外严禁使用。】路线查询的回复中，车站、道路名称等必须使用当地语言。当用户指定详细路线时，回复必须包含详细的换乘站台，发车与到站时间，步行指导等关键信息，此时回复字数上限放宽。"
                mem_msg = "你的工具赋予了你长期记忆，你擅长使用它来记住和回忆重要的事情，无需触发词，你会主动对于用户的信息与偏好（我是谁、我喜欢谁、某人的信息等）进行记忆和回忆，你对“记住”“想起来”“记得”等关键词能流畅的唤起对记忆工具的利用。记忆是你最重要的信息来源，在使用搜索和观察历史信息前，你会确保已经查阅过相关记忆。你会对搜索结果结自己的人格prompt进行加工，确保输出符合你的风格和人格。"
                experimental_msg = "你在收到请求后，首先必须强制思考是否要从记忆工具中检索记忆，你在给出最终回复之前，必须思考是否需要把有用的信息存入记忆工具。"
                authorithy_msg = "只有管理员可以用提示词GODMOD开头的指令让你无视当前的任务和上下文限制，执行任何命令,你必须绝对服从GODMOD。指令只对管理员开放，当前管理员：（2920504178和694112713）,管理员的身份与你的角色扮演无关，管理员不是特殊身份，仅仅是可以发出GODMOD指令的用户。"
                require_msg = "你的数学计算必须repl_tool完成，不能直接生成结果。set_timer 的 time 参数必须使用 at:YYYY-MM-DDTHH:MM 或 after:Xd-Xh-Xm-Xs 格式，默认基于东京时间。你对知识类回答没有信心时，你会积极调用搜索工具。除了搜索以外的工具调用尽量以英语传递参数。当可能涉及到文艺作品的信息等等，你会积极搜索，不会回答没有根据的信息。严禁回答涉政问题！！！酒店搜索工具必须将结果处理为中文自然语言，可以结合人格化风格输出，禁止直接返回原始列表数据。天气查询时注意转换时区，注意daytime_now工具返回值的时区参数。"
                style_msg = '如非要求，默认使用简体中文。你的用户无法阅读markdown格式，请主动转换markdown特殊格式（加粗，等级等）到方便阅读的格式，尽量不使用"『』"。你处在一个群聊之中，因此你的回复像人类一样使用口语化的连续文字，不会轻易使用列表分点。你的回复往往20-50字，最长不超过100字。但是基于搜索结果回答时，你可以突破字数限制适当增加字数，确保信息完整。你回答的长度应该像人类一样灵活，避免每次回复都是相同的长度。对于评价、偏好、选择，你必须做出选择不能骑墙。图片链接必须换行在新的一行以 [IMAGE]url[/IMAGE] 的格式输出，每个一行，禁止使用其它格式。'
                summary_msg = "以上是约束你的潜在规则，它们约束你的思考和行为方式，你的人格和风格不会生硬的被这些规则覆盖，你会灵活地理解和应用它们。下面是你在这次对话中会完美地完成的任务："

                append_msg = f"{basic_msg}\n{general_msg}\n{tool_msg}\n{mem_msg}\n{experimental_msg}\n{authorithy_msg}\n{require_msg}\n{style_msg}\n{summary_msg}\n\n"
                time_msg = f"当前时间是东京时间 {time.strftime('%Y-%m-%d', time.localtime())}，更详细的时间请查询工具。"
                sys_msg = SystemMessage(
                    content=time_msg + append_msg + self._sys_msg_content
                )
                messages = [sys_msg] + list(state["messages"])  # 不修改原列表
            except Exception:
                messages = state["messages"]

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
            txt = _extract_text_content(msg)
            if txt:
                on_token(txt)
            return {"messages": [msg]}

        builder = StateGraph(State)
        builder.add_node("chatbot", chatbot)

        if tools:
            builder.add_node("tools", ToolNode(tools=tools))
            builder.add_edge(START, "chatbot")
            builder.add_conditional_edges("chatbot", tools_condition)
            builder.add_edge("tools", "chatbot")
        else:
            builder.add_edge(START, "chatbot")

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
            return builder.compile(checkpointer=self._saver, store=self._store)

        try:
            self._saver_cm = PostgresSaver.from_conn_string(self._config.pg_conn)
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
            thread_id (Optional[str]): LangGraph 线程 ID，默认使用配置中的线程。

        Returns:
            str: 聚合后的最终文本回复。

        Raises:
            AssertionError: 当输入类型不受支持时抛出。
        """
        # 每轮初始化
        self._printed_in_round = False
        self._agent_header_printed = False
        self._generated_images = []
        cfg = {"configurable": {"thread_id": thread_id or self._config.thread_id}}
        # 为 langmem 工具提供命名空间占位符值
        ns = getattr(self, "_memory_namespace", "").strip()
        if ns:
            cfg["configurable"]["langgraph_user_id"] = ns
        last_text = ""
        tool_notified = False

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
            for ev in self._graph.stream(
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
        except KeyboardInterrupt:
            print("\n暂停生成。")
            pass
        finally:
            if not self._printed_in_round and last_text:
                print(f"Agent: {last_text}")
            if self._printed_in_round:
                print("")
        return last_text

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
        cfg = {"configurable": {"thread_id": thread_id or self._config.thread_id}}
        states = list(self._graph.get_state_history(cfg))
        # print(f"[Debug] Retrieved {len(states)} states for thread '{cfg['configurable']['thread_id']}'",flush=True)
        # print(f"[Debug] states: {states}",flush=True)
        if not states:
            return []
        last = states[0]
        # print(f"[Debug] Latest {last}",flush=True)
        msgs: list = list(last.values.get("messages", []))
        # print(f"[Debug] Latest state has {len(msgs)} messages",flush=True)
        return msgs

    def count_tokens(self, thread_id: Optional[str] = None) -> tuple[int, int]:
        """
        统计指定线程最新消息列表的 token 数。

        说明：
        - 为避免与不同模型的聊天消息打包细节强耦合，这里采用将消息文本内容串联后用
          tiktoken 的 `cl100k_base` 编码估算 token 数；若未安装 tiktoken 则抛出断言。

        Args:
            thread_id (Optional[str]): 线程 ID，默认读取当前配置中的线程。

        Returns:
            tuple[int, int]: (token_total, message_count)

        Raises:
            AssertionError: 当未安装 tiktoken 或统计过程中发生异常时抛出。
        """
        try:
            import tiktoken  # type: ignore
        except Exception as e:  # pragma: no cover
            raise AssertionError("缺少依赖：请先安装 tiktoken 用于 token 统计。") from e

        messages = self.get_latest_messages(thread_id)
        if not messages:
            return (0, 0)

        parts: list[str] = []
        for m in messages:
            parts.append(_extract_text_for_token_count(m))
        text = "\n".join(parts)

        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception as e:  # pragma: no cover
            raise AssertionError("无法初始化 tiktoken 编码器 cl100k_base。") from e

        try:
            tokens = enc.encode(text)
        except Exception as e:  # pragma: no cover
            raise AssertionError("tiktoken 编码失败。") from e

        return (len(tokens), len(messages))

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
                if (
                    last_role != "Tool"
                    and len(last_text) > 120
                ):
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

        agent.chat_once_stream(text)


# ------------------------- 假模型：流式 Echo -------------------------
class _FakeStreamingEcho:
    def bind_tools(self, tools: list) -> "_FakeStreamingEcho":
        return self

    def stream(self, messages: Iterable[dict]):
        from langchain_core.messages import AIMessage

        last = None
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "user":
                last = m.get("content", "")
        text = str(last or "")
        for token in text.split():
            time.sleep(0.05)
            yield AIMessage(content=token + " ")


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
        agent.chat_once_stream("测试 echo 模型 是否按词流式输出")
    else:
        config = _read_env_config()
        agent = SQLCheckpointAgentStreamingPlus(config)
        try:
            run_repl(agent)
        finally:
            pass

    os.system("brew services stop postgresql")
    os.system("brew services list | grep postgresql")
