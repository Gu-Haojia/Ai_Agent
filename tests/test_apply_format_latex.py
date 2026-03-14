"""
_apply_format 的 LaTeX 与 FORMAT 行为测试。
"""

from __future__ import annotations

import pytest

from sql_agent_cli_stream_plus import _apply_format


def test_apply_format_disabled_keeps_original_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    验证 FORMAT 未开启时返回原始文本。

    Args:
        monkeypatch (pytest.MonkeyPatch): 环境变量注入工具。

    Returns:
        None: 无返回值。
    """
    monkeypatch.setenv("FORMAT", "0")
    raw = r"**金额** $\\alpha$ 与 $$\\frac{1}{2}$$"
    assert _apply_format(raw) == raw


def test_apply_format_enabled_removes_markdown_and_parses_inline_latex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    验证 FORMAT 开启时会移除 Markdown 并替换行内公式。

    Args:
        monkeypatch (pytest.MonkeyPatch): 环境变量注入工具。

    Returns:
        None: 无返回值。
    """
    monkeypatch.setenv("FORMAT", "1")
    raw = r"结果是 **$\alpha + \beta$**"
    assert _apply_format(raw) == "结果是 α + β"


def test_apply_format_enabled_parses_display_and_multiple_formulas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    验证 FORMAT 开启时可处理多段公式并清除公式定界符。

    Args:
        monkeypatch (pytest.MonkeyPatch): 环境变量注入工具。

    Returns:
        None: 无返回值。
    """
    monkeypatch.setenv("FORMAT", "1")
    raw = r"A: $$\frac{1}{2}$$, B: $\sum_{i=1}^n i$"
    formatted = _apply_format(raw)
    assert formatted == "A: 1/2, B: ∑_i=1^n i"
    assert "$" not in formatted


def test_apply_format_enabled_skips_escaped_dollar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    验证转义美元符不会被误判为公式边界。

    Args:
        monkeypatch (pytest.MonkeyPatch): 环境变量注入工具。

    Returns:
        None: 无返回值。
    """
    monkeypatch.setenv("FORMAT", "1")
    raw = r"价格是 \$5，公式是 $\alpha$"
    assert _apply_format(raw) == r"价格是 \$5，公式是 α"
