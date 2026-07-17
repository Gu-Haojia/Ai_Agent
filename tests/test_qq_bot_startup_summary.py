"""QQ Bot 启动摘要渲染测试。"""

from __future__ import annotations

from datetime import datetime, timezone

from qq_group_bot import BotConfig, QQBotStartupSummary


def _summary() -> QQBotStartupSummary:
    """构造包含完整配置的启动摘要。

    Returns:
        QQBotStartupSummary: 测试使用的固定启动摘要。

    Raises:
        None: 本函数不主动抛出异常。
    """
    return QQBotStartupSummary(
        bot_config=BotConfig(
            host="0.0.0.0",
            port=8080,
            api_base="http://127.0.0.1:3001",
            secret="signature-secret",
            access_token="access-secret",
            allowed_groups=(10001, 10002),
            blacklist_groups=(20001,),
            limitlist_groups=(),
            cmd_allowed_users=(30001,),
            use_local_image_base64=True,
        ),
        model_name="google_genai:gemini-test",
        use_memory_checkpoint=False,
        thread_id="qq-default",
        thread_store=".qq_group_threads.json",
        memory_id="shared-memory",
        memory_store=".qq_group_memnames.json",
        prompt_name="default",
        image_directory="/tmp/qq-images/20260717~run",
        daily_time="09:00",
        daily_groups=(10001, 10002),
        nightly_time="21:00",
        nightly_groups=(),
        ticket_times=("12:00", "22:05"),
        ticket_groups=(10001,),
        restored_meru_tasks=3,
        restored_x_tasks=2,
        started_at=datetime(2026, 7, 17, 5, 32, 8, tzinfo=timezone.utc),
        startup_seconds=1.234,
    )


def test_startup_summary_renders_clear_runtime_state_without_secrets() -> None:
    """验证摘要展示关键状态但不泄露认证信息。

    Returns:
        None: 测试无返回值。

    Raises:
        None: 断言失败时由 pytest 报告。
    """
    output = _summary().render()

    assert "╭─ QQ Bot · ● READY" in output
    assert "http://0.0.0.0:8080 · 已监听" in output
    assert "http://127.0.0.1:3001 · 已配置，未探测" in output
    assert "Access Token ✓ · Signature ✓" in output
    assert "google_genai:gemini-test" in output
    assert "当前 Prompt   default" in output
    assert "检查点        PostgreSQL" in output
    assert "每日简报      09:00 · 2 个群" in output
    assert "晚间简报      关闭" in output
    assert "Ticket 检查   12:00、22:05 · 1 个群" in output
    assert "Meru 监控     已恢复 3 个" in output
    assert "X 监控        已恢复 2 个" in output
    assert "access-secret" not in output
    assert "signature-secret" not in output
    assert "\033[" not in output


def test_startup_summary_adds_color_only_when_requested() -> None:
    """验证 ANSI 颜色仅在显式启用时加入。

    Returns:
        None: 测试无返回值。

    Raises:
        None: 断言失败时由 pytest 报告。
    """
    output = _summary().render(use_color=True)

    assert "\033[1;92m● READY\033[0m" in output
    assert "\033[1;96m服务\033[0m" in output
    assert "access-secret" not in output
