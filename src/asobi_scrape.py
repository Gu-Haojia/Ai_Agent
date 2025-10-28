#!/usr/bin/env python3
"""Fetch ASOBI TICKET live/lottery information."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from html import unescape
from html.parser import HTMLParser
from typing import Any, Dict, Iterable, List, Optional

import requests

BASE_URL = "https://asobi-ticket.api.app.t-riple.com/api"
SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "codex-scraper/1.0",
        "Accept": "application/json",
    }
)


class HTMLStripper(HTMLParser):
    """Convert HTML into readable plain text."""

    def __init__(self) -> None:
        super().__init__()
        self._chunks: List[str] = []

    def handle_data(self, data: str) -> None:
        if data:
            self._chunks.append(data)

    def handle_br(self) -> None:  # pragma: no cover - HTMLParser hook
        self._chunks.append("\n")

    def handle_starttag(self, tag: str, attrs):  # pragma: no cover - HTMLParser hook
        if tag in {"p", "div", "li"}:
            self._chunks.append("\n")
        elif tag == "br":
            self._chunks.append("\n")

    def handle_endtag(self, tag: str):  # pragma: no cover - HTMLParser hook
        if tag in {"p", "div", "li"}:
            self._chunks.append("\n")

    def text(self) -> str:
        return unescape(" ".join(chunk.strip() for chunk in self._chunks if chunk).strip())


def html_to_text(html: Optional[str]) -> Optional[str]:
    if not html:
        return None
    parser = HTMLStripper()
    parser.feed(html)
    return parser.text().strip() or None


def iso_or_none(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        # Normalise to ISO8601 (strip microseconds if necessary).
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.isoformat()
    except ValueError:
        return value


def parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def get_json(path: str, *, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{BASE_URL}{path}"
    response = SESSION.get(url, params=params, timeout=15)
    response.raise_for_status()
    return response.json()


@dataclass
class Reception:
    id: str
    name: str
    status: str
    entry_type: str
    winning_type: Optional[str]
    entry_period_status: Optional[str]
    entry_period_starts_at: Optional[str]
    entry_period_ends_at: Optional[str]
    result_announcement_scheduled_at: Optional[str]
    deposit_period_starts_at: Optional[str]
    deposit_period_ends_at: Optional[str]
    stock_status: Optional[str]
    raw_html: Dict[str, Optional[str]] = field(default_factory=dict)


def parse_reception(raw: Dict[str, Any]) -> Reception:
    attrs = raw.get("attributes", {})
    return Reception(
        id=raw.get("id", ""),
        name=attrs.get("name", ""),
        status=attrs.get("status", ""),
        entry_type=attrs.get("entry_type", ""),
        winning_type=attrs.get("winning_type"),
        entry_period_status=attrs.get("entry_period_status"),
        entry_period_starts_at=iso_or_none(attrs.get("entry_period_starts_at")),
        entry_period_ends_at=iso_or_none(attrs.get("entry_period_ends_at")),
        result_announcement_scheduled_at=iso_or_none(attrs.get("result_announcement_scheduled_at")),
        deposit_period_starts_at=iso_or_none(attrs.get("deposit_period_starts_at")),
        deposit_period_ends_at=iso_or_none(attrs.get("deposit_period_ends_at")),
        stock_status=attrs.get("stock_status"),
        raw_html={
            "top_body": attrs.get("top_body"),
            "main_body": attrs.get("main_body"),
            "attention_body": attrs.get("attention_body"),
            "payment_method_description_body": attrs.get("payment_method_description_body"),
            "ticket_reception_method_description_body": attrs.get("ticket_reception_method_description_body"),
        },
    )


def fetch_receptions(slug: str) -> List[Reception]:
    payload = get_json("/v1/public/receptions", params={"booth_slug": slug})
    return [parse_reception(item) for item in payload.get("data", [])]


@dataclass
class Booth:
    slug: str
    name: str
    status: str
    entry_opening_receptions_count: int
    recent_reception_entry_period_starts_at: Optional[str]
    main_body_html: Optional[str]
    main_body_text: Optional[str]
    cover_image_urls: Optional[Dict[str, str]]
    receptions_open: List[Reception]
    receptions_upcoming: List[Reception]
    receptions_closed: List[Reception]


def collect_booths(per_page: int = 50) -> (List[Dict[str, Any]], Dict[str, Dict[str, Any]]):
    """Fetch all booth pages returning raw booths and included image lookup."""
    page = 1
    booths: List[Dict[str, Any]] = []
    images: Dict[str, Dict[str, Any]] = {}

    while True:
        payload = get_json("/v1/public/booths", params={"page": page, "per_page": per_page})
        data = payload.get("data", [])
        if not data:
            break
        booths.extend(data)

        for item in payload.get("included", []):
            if item.get("type") == "image":
                images[item["id"]] = item.get("attributes", {}).get("reading_urls", {})

        if len(data) < per_page:
            break
        page += 1

    return booths, images


def transform_booth(raw: Dict[str, Any], image_lookup: Dict[str, Dict[str, Any]]) -> Booth:
    attrs = raw.get("attributes", {})
    slug = attrs.get("slug", raw.get("id", ""))
    receptions = fetch_receptions(slug)

    open_receptions = [rec for rec in receptions if rec.entry_period_status == "within_entry_period"]
    upcoming_receptions = [rec for rec in receptions if rec.entry_period_status == "before_entry_period"]
    closed_receptions = [rec for rec in receptions if rec.entry_period_status not in {"within_entry_period", "before_entry_period"}]

    cover_image_id = (
        raw.get("relationships", {})
        .get("cover_image", {})
        .get("data", {})
        .get("id")
    )

    main_body = attrs.get("main_body")

    return Booth(
        slug=slug,
        name=attrs.get("name", ""),
        status=attrs.get("status", ""),
        entry_opening_receptions_count=attrs.get("entry_opening_receptions_count", 0),
        recent_reception_entry_period_starts_at=iso_or_none(attrs.get("recent_reception_entry_period_starts_at")),
        main_body_html=main_body,
        main_body_text=html_to_text(main_body),
        cover_image_urls=image_lookup.get(cover_image_id),
        receptions_open=open_receptions,
        receptions_upcoming=upcoming_receptions,
        receptions_closed=closed_receptions,
    )


def summarize(booths: Iterable[Booth]) -> str:
    lines: List[str] = []
    for booth in booths:
        lines.append(f"{booth.name} ({booth.slug}) — open: {len(booth.receptions_open)}, upcoming: {len(booth.receptions_upcoming)}")
        for rec in booth.receptions_open:
            lines.append(
                f"  [OPEN] {rec.name} ({rec.entry_type}) {rec.entry_period_starts_at} → {rec.entry_period_ends_at}"
            )
        for rec in booth.receptions_upcoming:
            lines.append(
                f"  [UPCOMING] {rec.name} ({rec.entry_type}) {rec.entry_period_starts_at} → {rec.entry_period_ends_at}"
            )
    return "\n".join(lines)


def export_active(booths: Iterable[Booth]) -> List[Dict[str, Any]]:
    active: List[Dict[str, Any]] = []
    for booth in booths:
        if not booth.receptions_open:
            continue
        active.append(
            {
                "booth_name": booth.name,
                "booth_slug": booth.slug,
                "main_body_text": booth.main_body_text,
                "open_receptions": [
                    {
                        "reception_name": rec.name,
                        "entry_period_starts_at": rec.entry_period_starts_at,
                        "entry_period_ends_at": rec.entry_period_ends_at,
                        "result_announcement_scheduled_at": rec.result_announcement_scheduled_at,
                    }
                    for rec in booth.receptions_open
                ],
            }
        )
    return active


def load_state(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"timestamp": None, "open_receptions": {}}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_state(path: str, receptions: Iterable[Reception], booth_map: Dict[str, str]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "open_receptions": {
            rec.id: {
                "booth_slug": booth_map.get(rec.id, ""),
                "entry_period_starts_at": rec.entry_period_starts_at,
            }
            for rec in receptions
        },
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def detect_new_open_receptions(
    booths: Iterable[Booth], previous_state: Dict[str, Any]
) -> List[Dict[str, Any]]:
    prev = previous_state.get("open_receptions", {})
    new_items: List[Dict[str, Any]] = []
    for booth in booths:
        for rec in booth.receptions_open:
            if rec.id not in prev:
                new_items.append(
                    {
                        "booth_name": booth.name,
                        "booth_slug": booth.slug,
                        "reception_name": rec.name,
                        "entry_period_starts_at": rec.entry_period_starts_at,
                        "entry_period_ends_at": rec.entry_period_ends_at,
                    }
                )
    return new_items


def build_reception_booth_map(booths: Iterable[Booth]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for booth in booths:
        for rec in booth.receptions_open:
            mapping[rec.id] = booth.slug
    return mapping


def remind_before_deadline(
    booths: Iterable[Booth], lead: timedelta
) -> List[Dict[str, Any]]:
    reminders: List[Dict[str, Any]] = []
    now = datetime.now(timezone.utc)
    threshold = now + lead
    for booth in booths:
        for rec in booth.receptions_open:
            end_dt = parse_datetime(rec.entry_period_ends_at)
            if not end_dt:
                continue
            if now < end_dt <= threshold:
                reminders.append(
                    {
                        "booth_name": booth.name,
                        "booth_slug": booth.slug,
                        "reception_name": rec.name,
                        "entry_period_ends_at": rec.entry_period_ends_at,
                        "remaining_hours": round((end_dt - now).total_seconds() / 3600, 2),
                    }
                )
    return reminders


def write_json(path: str, data: Any, pretty: bool = False) -> None:
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with (open(path, "w", encoding="utf-8") if path else os.fdopen(os.dup(1), "w", encoding="utf-8")) as fh:
        if pretty:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        else:
            json.dump(data, fh, ensure_ascii=False)
        fh.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape ASOBI TICKET live and lottery data.")
    parser.add_argument(
        "-o",
        "--output",
        default="asobi_data.json",
        help="Path to write JSON data (default: %(default)s). Use '-' for stdout only.",
    )
    parser.add_argument("--per-page", type=int, default=50, help="Number of booths per request (default: %(default)s).")
    parser.add_argument("--no-summary", action="store_true", help="Suppress console summary.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    parser.add_argument(
        "--export-active",
        help="Export current active (open) receptions with required fields to the given path.",
    )
    parser.add_argument(
        "--state-file",
        help="Path to store state for detecting newly opened receptions between runs.",
    )
    parser.add_argument(
        "--state-readonly",
        action="store_true",
        help="Use state file only for comparison without updating it.",
    )
    parser.add_argument(
        "--notify-new",
        action="store_true",
        help="When used with --state-file, print newly opened receptions since last run.",
    )
    parser.add_argument(
        "--remind-before",
        type=float,
        help="If set (hours), list open receptions ending within this time window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_booths, image_lookup = collect_booths(args.per_page)
    transformed = [transform_booth(raw, image_lookup) for raw in raw_booths]

    if not args.no_summary:
        print(summarize(transformed), flush=True)

    if args.export_active:
        active_payload = export_active(transformed)
        export_path = None if args.export_active == "-" else args.export_active
        write_json(export_path or "", active_payload, pretty=args.pretty)

    if args.state_file:
        previous_state = load_state(args.state_file)
        booth_map = build_reception_booth_map(transformed)
        open_receptions = [rec for booth in transformed for rec in booth.receptions_open]
        new_open = detect_new_open_receptions(transformed, previous_state)
        if args.notify_new:
            if new_open:
                print("新增抽選开启：")
                for item in new_open:
                    print(
                        f"- {item['booth_name']} / {item['reception_name']} "
                        f"开始：{item['entry_period_starts_at']} 截止：{item['entry_period_ends_at']}"
                    )
        else:
            print("没有新的抽選开启。")
        if not args.state_readonly:
            save_state(args.state_file, open_receptions, booth_map)

    if args.remind_before is not None:
        lead = timedelta(hours=args.remind_before)
        reminders = remind_before_deadline(transformed, lead)
        if reminders:
            print(f"抽選即将在 {args.remind_before} 小时内截止：")
            for item in reminders:
                print(
                    f"- {item['booth_name']} / {item['reception_name']} "
                    f"截止：{item['entry_period_ends_at']} 剩余约 {item['remaining_hours']} 小时"
                )
        else:
            print(f"未来 {args.remind_before} 小时内没有即将截止的抽選。")

    payload = [
        {
            **{
                "slug": booth.slug,
                "name": booth.name,
                "status": booth.status,
                "entry_opening_receptions_count": booth.entry_opening_receptions_count,
                "recent_reception_entry_period_starts_at": booth.recent_reception_entry_period_starts_at,
                "main_body_html": booth.main_body_html,
                "main_body_text": booth.main_body_text,
                "cover_image_urls": booth.cover_image_urls,
            },
            "receptions": {
                "open": [asdict(rec) for rec in booth.receptions_open],
                "upcoming": [asdict(rec) for rec in booth.receptions_upcoming],
                "closed": [asdict(rec) for rec in booth.receptions_closed],
            },
        }
        for booth in transformed
    ]

    output_path = None if args.output == "-" else args.output
    write_json(output_path or "", payload, pretty=args.pretty)


if __name__ == "__main__":
    main()
