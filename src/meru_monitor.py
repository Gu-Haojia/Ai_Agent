"""
Meru 关键词搜索与监控模块。

基于 Mercari DPoP API 提供一次性查询与后台监控能力，便于 QQ 机器人复用。
"""

from __future__ import annotations

import base64
import time
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Event, Lock, Thread
from typing import Callable, Optional, Sequence

import jwt
import requests
from cryptography.hazmat.primitives.asymmetric import ec

API_URL = "https://api.mercari.jp/v2/entities:search"
ITEM_URL = "https://jp.mercari.com/item/{id}"
DEFAULT_LIMIT = 5

# 静默掉 macOS LibreSSL 的兼容性告警
warnings.filterwarnings(
    "ignore",
    message="urllib3 v2 only supports OpenSSL",
    category=Warning,
)


def _b64url_int(value: int) -> str:
    """
    将整数转换为去掉填充的 URL-safe Base64 字符串。

    Args:
        value (int): 需要编码的整数。

    Returns:
        str: 编码后的字符串。
    """
    width = (value.bit_length() + 7) // 8
    raw = value.to_bytes(width, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _safe_int(value: object) -> Optional[int]:
    """
    尝试将输入转换为整数，失败时返回 None。

    Args:
        value (object): 待转换的值。

    Returns:
        Optional[int]: 成功时返回整数，否则为 None。
    """
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:
        return None


def _format_ts(ts: object) -> str:
    """
    将时间戳转换为本地时间字符串。

    Args:
        ts (object): Unix 时间戳（秒）。

    Returns:
        str: 格式化后的时间，失败时返回空字符串。
    """
    try:
        unix_ts = int(ts)  # type: ignore[arg-type]
    except Exception:
        return ""
    dt = datetime.fromtimestamp(unix_ts, tz=timezone.utc).astimezone()
    return dt.strftime("%m-%d %H:%M")


@dataclass(frozen=True)
class MeruSearchResult:
    """
    Meru 搜索结果。

    Attributes:
        keyword (str): 查询关键词。
        item_id (str): 商品 ID。
        name (str): 商品标题。
        price (Optional[int]): 价格（单位日元），未知时为 None。
        created_label (str): 创建时间字符串。
        url (str): 商品链接。
    """

    keyword: str
    item_id: str
    name: str
    price: Optional[int]
    created_label: str
    url: str

    def to_line(self, prefix: str = "") -> str:
        """
        格式化为单行字符串。

        Args:
            prefix (str): 行前缀，用于区分状态。

        Returns:
            str: 格式化后的单行文本。
        """
        price_part = f"¥{self.price}" if self.price is not None else "价格未知"
        pieces = [
            prefix.strip(),
            f"[{self.keyword}]",
            self.name or "(无标题)",
            price_part,
        ]
        if self.created_label:
            pieces.append(self.created_label)
        line = " | ".join(piece for piece in pieces if piece)
        if self.url:
            line = f"{line} | {self.url}"
        return line


class MercariDPoPClient:
    """
    简化版 Mercari DPoP 客户端，用于执行搜索请求。
    """

    def __init__(self) -> None:
        """初始化并生成 EC 私钥。"""
        self._key = ec.generate_private_key(ec.SECP256R1())

    def _build_dpop(self, method: str, url: str) -> str:
        """
        构造 DPoP JWT。

        Args:
            method (str): HTTP 方法。
            url (str): 完整 URL。

        Returns:
            str: DPoP header 字符串。
        """
        public_numbers = self._key.public_key().public_numbers()
        jwk = {
            "kty": "EC",
            "crv": "P-256",
            "x": _b64url_int(public_numbers.x),
            "y": _b64url_int(public_numbers.y),
        }
        headers = {"typ": "dpop+jwt", "alg": "ES256", "jwk": jwk}
        payload = {
            "htu": url,
            "htm": method,
            "iat": int(time.time()),
            "jti": str(uuid.uuid4()),
        }
        return jwt.encode(payload, self._key, algorithm="ES256", headers=headers)

    def search(
        self,
        keyword: str,
        status_on_sale_only: bool = True,
        page_size: int = 40,
    ) -> dict:
        """
        调用 Mercari 搜索接口。

        Args:
            keyword (str): 搜索关键词。
            status_on_sale_only (bool): True 时仅返回在售商品。
            page_size (int): 单页大小。

        Returns:
            dict: 接口返回的 JSON。
        """
        assert keyword and keyword.strip(), "keyword 不能为空"
        dpop = self._build_dpop("POST", API_URL)
        statuses = ["STATUS_ON_SALE"] if status_on_sale_only else []
        body = {
            "userId": "",
            "pageSize": page_size,
            "pageToken": "",
            "searchSessionId": uuid.uuid4().hex,
            "source": "BaseSerp",
            "indexRouting": "INDEX_ROUTING_UNSPECIFIED",
            "thumbnailTypes": [],
            "searchCondition": {
                "keyword": keyword,
                "excludeKeyword": "",
                "sort": "SORT_CREATED_TIME",
                "order": "ORDER_DESC",
                "status": statuses,
                "sizeId": [],
                "categoryId": [],
                "brandId": [],
                "sellerId": [],
                "priceMin": 0,
                "priceMax": 0,
                "itemConditionId": [],
                "shippingPayerId": [],
                "shippingFromArea": [],
                "shippingMethod": [],
                "colorId": [],
                "hasCoupon": False,
                "createdAfterDate": None,
                "createdBeforeDate": None,
                "attributes": [],
                "itemTypes": [],
                "skuIds": [],
                "shopIds": [],
                "excludeShippingMethodIds": [],
            },
            "serviceFrom": "web-suruga-ssr",
            "withItemBrand": True,
            "withItemSize": False,
            "withItemPromotions": True,
            "withItemSizes": True,
            "withShopname": False,
            "useDynamicAttribute": True,
            "withSuggestedItems": True,
            "withOfferPricePromotion": True,
            "withProductSuggest": True,
            "withParentProducts": False,
            "withProductArticles": True,
            "withSearchConditionId": False,
            "withAuction": True,
            "laplaceDeviceUuid": "",
        }
        headers = {
            "User-Agent": "MercariWatcher/1.0 (+https://jp.mercari.com)",
            "Content-Type": "application/json",
            "X-Platform": "web",
            "X-Country-Code": "JP",
            "DPoP": dpop,
            "Accept-Language": "ja",
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://jp.mercari.com/",
        }
        response = requests.post(API_URL, json=body, headers=headers, timeout=20)
        response.raise_for_status()
        return response.json()


class _MeruWatchTask:
    """
    内部监控任务，负责轮询并发现新品。
    """

    def __init__(
        self,
        keyword: str,
        interval: float,
        limit_per_cycle: int,
        price_threshold: Optional[int],
        fetcher: Callable[[], Sequence[MeruSearchResult]],
        formatter: Callable[[Sequence[MeruSearchResult], str], str],
        notify: Callable[[str], None],
        notify_price: Optional[Callable[[str], None]],
    ) -> None:
        """
        初始化监控任务。

        Args:
            keyword (str): 监控关键词。
            interval (float): 轮询间隔秒数。
            limit_per_cycle (int): 单轮最多推送条目数。
            price_threshold (Optional[int]): 价格提醒阈值。
            fetcher (Callable[[], Sequence[MeruSearchResult]]): 数据获取函数。
            formatter (Callable[[Sequence[MeruSearchResult], str], str]): 结果格式化函数。
            notify (Callable[[str], None]): 新品通知回调。
            notify_price (Optional[Callable[[str], None]]): 价格提醒回调。
        """
        assert interval > 0, "interval 必须大于 0"
        assert limit_per_cycle > 0, "limit_per_cycle 必须大于 0"
        self._keyword = keyword
        self._interval = interval
        self._limit = limit_per_cycle
        self._price_threshold = price_threshold
        self._fetcher = fetcher
        self._formatter = formatter
        self._notify = notify
        self._notify_price = notify_price
        self._stop_event = Event()
        self._thread = Thread(target=self._run, name="meru-watch", daemon=True)
        self._seen: set[str] = set()
        self._first_cycle = True

    def start(self) -> None:
        """启动后台线程。"""
        self._thread.start()

    def stop(self) -> None:
        """请求停止并等待线程退出。"""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self._interval + 1)

    def alive(self) -> bool:
        """
        判断线程是否仍在运行。

        Returns:
            bool: True 表示线程存活。
        """
        return self._thread.is_alive()

    def _run(self) -> None:
        """轮询获取数据并推送新品通知。"""
        while not self._stop_event.is_set():
            try:
                results = list(self._fetcher())
                if self._first_cycle:
                    for item in results:
                        self._seen.add(item.item_id)
                    self._first_cycle = False
                else:
                    new_items = [
                        item for item in results if item.item_id not in self._seen
                    ]
                    if new_items:
                        self._handle_new_items(new_items)
                    for item in results:
                        self._seen.add(item.item_id)
                    # 控制 seen 长度，避免无限增长
                    if len(self._seen) > 400:
                        self._seen = set(item.item_id for item in results)
            except Exception as err:
                self._notify(f"[MeruWatch] 监控失败：{err}")
            self._stop_event.wait(self._interval)

    def _handle_new_items(self, items: Sequence[MeruSearchResult]) -> None:
        """
        处理新品并触发通知。

        Args:
            items (Sequence[MeruSearchResult]): 新发现的商品列表。
        """
        subset = list(items)[: self._limit]
        message = self._formatter(subset, "NEW")
        self._notify(message)
        if self._price_threshold is None or not self._notify_price:
            return
        affordable = [
            item
            for item in subset
            if item.price is not None and item.price <= self._price_threshold
        ]
        if not affordable:
            return
        price_msg = self._formatter(
            affordable, f"PRICE<= {self._price_threshold}"
        )
        self._notify_price(price_msg)


class MeruMonitorManager:
    """
    Meru 搜索与监控管理器，提供一次查询与单任务监控能力。
    """

    def __init__(self, client: Optional[MercariDPoPClient] = None) -> None:
        """
        初始化管理器。

        Args:
            client (Optional[MercariDPoPClient]): 可选的自定义客户端。
        """
        self._client = client or MercariDPoPClient()
        self._lock = Lock()
        self._watch_task: Optional[_MeruWatchTask] = None

    def search(self, keyword: str, limit: int = DEFAULT_LIMIT) -> list[MeruSearchResult]:
        """
        执行一次关键词查询。

        Args:
            keyword (str): 搜索关键词。
            limit (int): 返回的最大条目数。

        Returns:
            list[MeruSearchResult]: 转换后的结果列表（按时间倒序）。
        """
        limit_value = max(1, limit) if limit else DEFAULT_LIMIT
        results = self._fetch_results(keyword)
        return results[:limit_value]

    def start_watch(
        self,
        keyword: str,
        interval: float,
        limit_per_cycle: int = DEFAULT_LIMIT,
        price_threshold: Optional[int] = None,
        notify: Optional[Callable[[str], None]] = None,
        notify_price: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        启动新品监控（全局仅允许一个监控任务）。

        Args:
            keyword (str): 监控关键词。
            interval (float): 轮询间隔秒数。
            limit_per_cycle (int): 单轮推送上限。
            price_threshold (Optional[int]): 价格提醒阈值。
            notify (Optional[Callable[[str], None]]): 新品通知回调。
            notify_price (Optional[Callable[[str], None]]): 价格提醒回调。

        Raises:
            RuntimeError: 当已有监控任务在运行时抛出。
            AssertionError: 参数校验失败时抛出。
        """
        assert keyword and keyword.strip(), "keyword 不能为空"
        assert interval > 0, "interval 必须大于 0"
        assert limit_per_cycle > 0, "limit_per_cycle 必须大于 0"
        assert notify is not None, "notify 回调不能为空"
        with self._lock:
            if self._watch_task and self._watch_task.alive():
                raise RuntimeError("已有监控任务在运行，请先关闭后再启动。")
            fetcher = lambda: self._fetch_results(keyword)
            formatter = lambda items, tag: self.format_lines(items, tag)
            task = _MeruWatchTask(
                keyword=keyword.strip(),
                interval=interval,
                limit_per_cycle=limit_per_cycle,
                price_threshold=price_threshold,
                fetcher=fetcher,
                formatter=formatter,
                notify=notify,
                notify_price=notify_price,
            )
            self._watch_task = task
            task.start()

    def stop_watch(self) -> bool:
        """
        停止当前监控任务。

        Returns:
            bool: True 表示存在任务且已请求停止，False 表示无任务。
        """
        with self._lock:
            task = self._watch_task
            if not task:
                return False
            task.stop()
            self._watch_task = None
            return True

    def _fetch_results(self, keyword: str) -> list[MeruSearchResult]:
        """
        调用接口并转换为标准结果。

        Args:
            keyword (str): 搜索关键词。

        Returns:
            list[MeruSearchResult]: 转换后的结果。
        """
        data = self._client.search(keyword)
        items = data.get("items") or []
        results: list[MeruSearchResult] = []
        for item in items:
            item_id = str(item.get("id") or "").strip()
            if not item_id:
                continue
            price = _safe_int(item.get("price"))
            name = str(item.get("name") or "").strip()
            created_label = _format_ts(item.get("created"))
            url = ITEM_URL.format(id=item_id)
            results.append(
                MeruSearchResult(
                    keyword=keyword,
                    item_id=item_id,
                    name=name,
                    price=price,
                    created_label=created_label,
                    url=url,
                )
            )
        return results

    @staticmethod
    def format_lines(
        items: Sequence[MeruSearchResult],
        tag: str,
    ) -> str:
        """
        将结果序列格式化为多行文本。

        Args:
            items (Sequence[MeruSearchResult]): 需要格式化的商品列表。
            tag (str): 标签前缀，用于标识状态。

        Returns:
            str: 合并后的多行字符串。
        """
        lines = []
        if not items:
            return ""
        keyword = items[0].keyword
        header = f"[{tag}]"
        if keyword:
            header = f"{header} | [{keyword}]"
        lines.append(header)
        for idx, item in enumerate(items, 1):
            price_part = f"¥{item.price}" if item.price is not None else "价格未知"
            parts = [f"#{idx}", item.name or "(无标题)", price_part]
            if item.created_label:
                parts.append(item.created_label)
            if item.url:
                parts.append(item.url)
            lines.append(" | ".join(parts))
        return "\n".join(lines)
