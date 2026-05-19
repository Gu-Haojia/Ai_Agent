"""
Export a Playwright storage_state file after an interactive X login.

Run this on a machine that can open a headed browser. The generated JSON file
contains login cookies and must be treated like a password.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Open X in a browser and export Playwright storage_state."
    )
    parser.add_argument(
        "--output",
        default="secret/x_storage_state.json",
        help="Path for the exported storage_state JSON.",
    )
    parser.add_argument(
        "--profile-dir",
        default=".x-login-profile",
        help="Temporary local browser profile directory for the login session.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Open a headed browser, wait for manual login, and save storage_state.
    """
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    profile_dir = Path(args.profile_dir).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit(
            "缺少 playwright。请先运行：python -m pip install playwright && "
            "python -m playwright install chromium"
        ) from exc

    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=False,
            viewport={"width": 1280, "height": 900},
        )
        try:
            page = context.pages[0] if context.pages else context.new_page()
            page.goto("https://x.com/home", wait_until="domcontentloaded")
            print("请在弹出的浏览器里登录 X。")
            print("登录完成并能看到首页后，回到这个终端按 Enter 保存登录态。")
            input()
            context.storage_state(path=str(output))
            print(f"已保存登录态：{output}")
        finally:
            context.close()


if __name__ == "__main__":
    main()
