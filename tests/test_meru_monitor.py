"""
Meru 监控任务的单元测试。
"""

from __future__ import annotations

import unittest
from typing import Sequence

from src.meru_monitor import MeruMonitorManager, MeruSearchResult, _MeruWatchTask

EventRecord = tuple[str, str, tuple[str, ...], str]


def _make_item(
    item_id: str,
    price: int | None,
    previous_price: int | None = None,
) -> MeruSearchResult:
    """
    构造测试用 Meru 商品结果。

    Args:
        item_id (str): 商品 ID。
        price (int | None): 当前价格。
        previous_price (int | None): 降价前价格。

    Returns:
        MeruSearchResult: 测试商品结果。
    """
    return MeruSearchResult(
        keyword="watch",
        item_id=item_id,
        name=f"Item {item_id}",
        price=price,
        previous_price=previous_price,
        created_label="05-24 12:00",
        url=f"https://example.com/{item_id}",
        image_urls=(f"https://example.com/{item_id}.jpg",),
    )


class MeruPriceDropWatchTests(unittest.TestCase):
    """
    验证 Meru 监控对旧商品降价的处理。
    """

    def _build_task(
        self,
        events: list[EventRecord],
        price_threshold: int | None = None,
        price_only: bool = False,
    ) -> _MeruWatchTask:
        """
        构造不启动线程的监控任务。

        Args:
            events (list[EventRecord]): 收集通知事件的列表。
            price_threshold (int | None): 价格提醒阈值。
            price_only (bool): 是否只发送低价提醒。

        Returns:
            _MeruWatchTask: 测试用监控任务。
        """

        def _formatter(items: Sequence[MeruSearchResult], tag: str) -> str:
            """
            使用生产格式化逻辑生成消息文本。

            Args:
                items (Sequence[MeruSearchResult]): 商品列表。
                tag (str): 消息标签。

            Returns:
                str: 格式化后的消息文本。
            """
            return MeruMonitorManager.format_lines(items, tag)

        def _notify(text: str) -> None:
            """
            记录普通文本通知。

            Args:
                text (str): 通知文本。

            Returns:
                None: 无返回值。
            """
            events.append(("text", text, (), ""))

        def _notify_price(text: str) -> None:
            """
            记录价格文本通知。

            Args:
                text (str): 通知文本。

            Returns:
                None: 无返回值。
            """
            events.append(("price_text", text, (), ""))

        def _notify_media(
            text: str, items: Sequence[MeruSearchResult], tag: str
        ) -> None:
            """
            记录普通图文通知。

            Args:
                text (str): 通知文本。
                items (Sequence[MeruSearchResult]): 通知商品。
                tag (str): 消息标签。

            Returns:
                None: 无返回值。
            """
            events.append(
                ("media", text, tuple(item.item_id for item in items), tag)
            )

        def _notify_price_media(
            text: str, items: Sequence[MeruSearchResult], tag: str
        ) -> None:
            """
            记录价格图文通知。

            Args:
                text (str): 通知文本。
                items (Sequence[MeruSearchResult]): 通知商品。
                tag (str): 消息标签。

            Returns:
                None: 无返回值。
            """
            events.append(
                ("price_media", text, tuple(item.item_id for item in items), tag)
            )

        return _MeruWatchTask(
            keyword="watch",
            interval=60,
            limit_per_cycle=5,
            price_threshold=price_threshold,
            price_only=price_only,
            group_id=1,
            user_id=2,
            fetcher=lambda: [],
            formatter=_formatter,
            notify=_notify,
            notify_price=_notify_price if price_threshold is not None else None,
            notify_media=_notify_media,
            notify_price_media=(
                _notify_price_media if price_threshold is not None else None
            ),
        )

    def test_collect_price_drop_items_records_previous_price(self) -> None:
        """
        已见商品降价时，应携带上一轮价格。
        """
        events: list[EventRecord] = []
        task = self._build_task(events)
        task._seen.add("item-1")
        task._last_prices["item-1"] = 6000

        drops = task._collect_price_drop_items([_make_item("item-1", 4500)])

        self.assertEqual(len(drops), 1)
        self.assertEqual(drops[0].previous_price, 6000)

    def test_price_drop_below_threshold_uses_price_media_channel(self) -> None:
        """
        降价后低于阈值时，应复用价格图文通道以触发 @。
        """
        events: list[EventRecord] = []
        task = self._build_task(events, price_threshold=5000)

        task._handle_price_drop_items(
            [_make_item("item-1", 4800, previous_price=6000)]
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][0], "price_media")
        self.assertEqual(events[0][2], ("item-1",))
        self.assertEqual(events[0][3], "PRICE_DROP<= 5000")
        self.assertIn("原价：¥6000", events[0][1])

    def test_price_drop_above_threshold_uses_normal_media_channel(self) -> None:
        """
        降价后仍高于阈值时，应只发送普通降价通知。
        """
        events: list[EventRecord] = []
        task = self._build_task(events, price_threshold=5000)

        task._handle_price_drop_items(
            [_make_item("item-1", 7000, previous_price=8000)]
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][0], "media")
        self.assertEqual(events[0][2], ("item-1",))
        self.assertEqual(events[0][3], "PRICE_DROP")
        self.assertIn("[PRICE DROP] | [watch]", events[0][1])
        self.assertNotIn("低于", events[0][1])

    def test_price_only_ignores_drop_above_threshold(self) -> None:
        """
        仅低价模式下，高于阈值的降价不应发送普通通知。
        """
        events: list[EventRecord] = []
        task = self._build_task(events, price_threshold=5000, price_only=True)

        task._handle_price_drop_items(
            [_make_item("item-1", 7000, previous_price=8000)]
        )

        self.assertEqual(events, [])

    def test_record_current_items_prevents_repeated_same_price_drop(self) -> None:
        """
        降价处理后更新基准价格，同价位不应重复触发。
        """
        events: list[EventRecord] = []
        task = self._build_task(events)
        task._seen.add("item-1")
        task._last_prices["item-1"] = 6000
        current_items = [_make_item("item-1", 5000)]

        drops = task._collect_price_drop_items(current_items)
        task._record_current_items(current_items)
        repeated = task._collect_price_drop_items(current_items)

        self.assertEqual(len(drops), 1)
        self.assertEqual(repeated, [])


if __name__ == "__main__":
    unittest.main()
