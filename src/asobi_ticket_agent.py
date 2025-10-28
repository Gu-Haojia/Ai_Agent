"""
ASOBI 抽選查询服务。

封装原有 asobi_scrape 数据抓取逻辑，提供面向 Agent 的查询接口。
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Iterable, List

from src.asobi_scrape import (
    Booth,
    collect_booths,
    load_state,
    save_state,
    transform_booth,
    build_reception_booth_map,
)


class AsobiTicketQuery:
    """
    ASOBI 抽選查询类，支持 list、update、check 三种模式。

    Attributes:
        per_page (int): 单次请求的展位数量。
        state_path (str): 持久化文件路径，仅在 update 模式更新。
    """

    def __init__(
        self,
        per_page: int = 50,
        state_dir: str = "ticket_data",
        state_filename: str = "asobi_state.json",
    ) -> None:
        """
        初始化查询实例。

        Args:
            per_page (int): API 单页抓取数量，默认 50。
            state_dir (str): 持久化目录，默认 ``ticket_data``。
            state_filename (str): 状态文件名称，默认 ``asobi_state.json``。

        Raises:
            AssertionError: 当 per_page 小于 1 或参数为空时抛出。
        """
        assert isinstance(per_page, int) and per_page > 0, "per_page 必须为正整数"
        assert isinstance(state_dir, str) and state_dir.strip(), "state_dir 不能为空"
        assert isinstance(state_filename, str) and state_filename.strip(), "state_filename 不能为空"
        self._per_page = per_page
        self._state_dir = state_dir
        self._state_path = os.path.join(state_dir, state_filename)

    def run(self, mode: str) -> str:
        """
        执行指定模式的查询。

        Args:
            mode (str): 查询模式，支持 ``list``、``update``、``check``。

        Returns:
            str: JSON 字符串形式的返回结果。

        Raises:
            AssertionError: 当 mode 非法时抛出。
        """
        assert isinstance(mode, str) and mode.strip(), "mode 不能为空"
        normalized = mode.strip().lower()
        booths = self._fetch_booths()
        if normalized == "list":
            return self._handle_list(booths)
        if normalized == "update":
            return self._handle_update(booths)
        if normalized == "check":
            return self._handle_check(booths)
        raise AssertionError("mode 仅支持 list / update / check")

    def _fetch_booths(self) -> List[Booth]:
        """
        抓取并转换全部展位数据。

        Returns:
            list[Booth]: 转换后的展位集合。
        """
        raw_booths, image_lookup = collect_booths(self._per_page)
        return [transform_booth(raw, image_lookup) for raw in raw_booths]

    def _handle_list(self, booths: Iterable[Booth]) -> str:
        """
        列出所有抽選信息。

        Args:
            booths (Iterable[Booth]): 展位数据迭代器。

        Returns:
            str: JSON 字符串，字段与旧脚本保持一致。
        """
        payload: List[dict[str, Any]] = []
        for booth in booths:
            entry = self._build_booth_payload(booth)
            if entry:
                payload.append(entry)
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _handle_update(self, booths: Iterable[Booth]) -> str:
        """
        更新并返回最新开启的抽選。

        Args:
            booths (Iterable[Booth]): 展位数据迭代器。

        Returns:
            str: JSON 字符串，包含 ``updated`` 与 ``new_open`` 字段。
        """
        previous_state = self._load_state()
        new_mapping = self._collect_new_open(booths, previous_state)
        open_receptions = [rec for booth in booths for rec in booth.receptions_open]
        booth_map = build_reception_booth_map(booths)
        # 仅 update 模式写入持久化
        os.makedirs(self._state_dir, exist_ok=True)
        save_state(self._state_path, open_receptions, booth_map)
        result: dict[str, Any] = {
            "updated": any(new_mapping.values()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "new_open": [],
        }
        if any(new_mapping.values()):
            entries: List[dict[str, Any]] = []
            for booth in booths:
                rec_ids = set(new_mapping.get(booth.slug, []))
                if not rec_ids:
                    continue
                entry = self._build_booth_payload(booth, rec_ids)
                if entry:
                    entries.append(entry)
            result["new_open"] = entries
        return json.dumps(result, ensure_ascii=False, indent=2)

    def _handle_check(self, booths: Iterable[Booth]) -> str:
        """
        检查是否存在潜在更新，但不修改持久化。

        Args:
            booths (Iterable[Booth]): 展位数据迭代器。

        Returns:
            str: JSON 字符串，字段 ``has_update`` 表示可能有新增抽選。
        """
        previous_state = self._load_state()
        new_mapping = self._collect_new_open(booths, previous_state)
        result = {
            "has_update": any(new_mapping.values()),
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

    def _load_state(self) -> dict[str, Any]:
        """
        读取持久化状态，若文件缺失或损坏则返回默认结构。

        Returns:
            dict[str, Any]: 状态字典。
        """
        try:
            return load_state(self._state_path)
        except json.JSONDecodeError:
            return {"timestamp": None, "open_receptions": {}}

    def _collect_new_open(
        self,
        booths: Iterable[Booth],
        previous_state: dict[str, Any],
    ) -> dict[str, list[str]]:
        """
        统计相对于历史状态新增的开放抽選。

        Args:
            booths (Iterable[Booth]): 当前展位数据。
            previous_state (dict[str, Any]): 历史状态。

        Returns:
            dict[str, list[str]]: 按展位 slug 归类的新增抽選 ID。
        """
        prev_ids = set((previous_state.get("open_receptions") or {}).keys())
        new_mapping: dict[str, list[str]] = {}
        for booth in booths:
            for rec in booth.receptions_open:
                if rec.id not in prev_ids:
                    new_mapping.setdefault(booth.slug, []).append(rec.id)
        return new_mapping

    def _build_booth_payload(
        self,
        booth: Booth,
        only_ids: set[str] | None = None,
    ) -> dict[str, Any] | None:
        """
        构建展位输出结构。

        Args:
            booth (Booth): 展位对象。
            only_ids (set[str] | None): 若提供，仅包含这些抽選 ID。

        Returns:
            dict[str, Any] | None: 展位输出，若无符合抽選则返回 None。
        """
        open_payload: list[dict[str, Any]] = []
        for rec in booth.receptions_open:
            if only_ids is not None and rec.id not in only_ids:
                continue
            open_payload.append(
                {
                    "reception_name": rec.name,
                    "entry_period_starts_at": rec.entry_period_starts_at,
                    "entry_period_ends_at": rec.entry_period_ends_at,
                    "result_announcement_scheduled_at": rec.result_announcement_scheduled_at,
                }
            )
        if not open_payload:
            return None
        return {
            "booth_name": booth.name,
            "main_body_text": booth.main_body_text,
            "open_receptions": open_payload,
        }
