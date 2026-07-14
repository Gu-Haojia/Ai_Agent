"""NapCat Prompt 账号资料管理与 API 调用测试。"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest import mock

import pytest

import qq_group_bot
from src.napcat_account_profile import PromptAccountProfileManager


def test_manager_creates_missing_local_config(tmp_path: Path) -> None:
    """验证初始化时自动创建头像目录与空配置。

    Args:
        tmp_path (Path): pytest 临时目录。

    Returns:
        None: 测试通过时无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    avatar_dir = tmp_path / "prompts" / "avatars"
    config_path = avatar_dir / "account_profiles.json"

    manager = PromptAccountProfileManager(config_path, avatar_dir)

    assert avatar_dir.is_dir()
    assert config_path.read_text(encoding="utf-8") == "{}\n"
    assert manager.resolve("default") is None


def test_manager_resolves_chinese_profile_and_avatar(tmp_path: Path) -> None:
    """验证中文 Prompt、昵称和头像文件名能够正确解析。

    Args:
        tmp_path (Path): pytest 临时目录。

    Returns:
        None: 测试通过时无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    avatar_dir = tmp_path / "avatars"
    avatar_dir.mkdir()
    avatar_bytes = b"fake-avatar"
    (avatar_dir / "藤田ことね.png").write_bytes(avatar_bytes)
    config_path = tmp_path / "account_profiles.json"
    config_path.write_text(
        json.dumps(
            {
                "藤田ことね": {
                    "nickname": "藤田ことね",
                    "avatar_file": "藤田ことね.png",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manager = PromptAccountProfileManager(config_path, avatar_dir)

    profile = manager.resolve("藤田ことね")

    assert profile is not None
    assert profile.nickname == "藤田ことね"
    assert profile.avatar_file == "藤田ことね.png"
    assert profile.avatar_base64 == base64.b64encode(avatar_bytes).decode("ascii")


def test_manager_returns_none_for_untracked_prompt(tmp_path: Path) -> None:
    """验证未登记 Prompt 不要求头像目录且返回 ``None``。

    Args:
        tmp_path (Path): pytest 临时目录。

    Returns:
        None: 测试通过时无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    config_path = tmp_path / "account_profiles.json"
    config_path.write_text("{}", encoding="utf-8")
    manager = PromptAccountProfileManager(config_path, tmp_path / "avatars")

    assert manager.resolve("default") is None


@pytest.mark.parametrize(
    "avatar_file",
    ["../avatar.png", "sub/avatar.png", "sub\\avatar.png"],
)
def test_manager_rejects_avatar_path(
    tmp_path: Path,
    avatar_file: str,
) -> None:
    """验证配置只能填写头像文件名，不能填写路径。

    Args:
        tmp_path (Path): pytest 临时目录。
        avatar_file (str): 非法头像路径。

    Returns:
        None: 测试通过时无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    config_path = tmp_path / "account_profiles.json"
    config_path.write_text(
        json.dumps(
            {
                "default": {
                    "nickname": "默认昵称",
                    "avatar_file": avatar_file,
                }
            }
        ),
        encoding="utf-8",
    )
    manager = PromptAccountProfileManager(config_path, tmp_path / "avatars")

    with pytest.raises(AssertionError, match="只允许填写文件名"):
        manager.resolve("default")


def test_onebot_action_sends_authorized_json_and_accepts_success() -> None:
    """验证 OneBot action 请求结构和成功业务响应。

    Returns:
        None: 测试通过时无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    response = mock.MagicMock()
    response.__enter__.return_value.status = 200
    response.__enter__.return_value.read.return_value = json.dumps(
        {"status": "ok", "retcode": 0, "data": {}}
    ).encode("utf-8")

    with mock.patch.object(qq_group_bot, "urlopen", return_value=response) as opener:
        result = qq_group_bot._call_onebot_action(
            "http://onebot",
            "set_qq_profile",
            {"nickname": "藤田ことね"},
            "token",
        )

    assert result["status"] == "ok"
    request = opener.call_args.args[0]
    assert request.full_url == "http://onebot/set_qq_profile"
    assert request.headers["Authorization"] == "Bearer token"
    assert json.loads(request.data.decode("utf-8")) == {"nickname": "藤田ことね"}


def test_onebot_action_rejects_http_200_business_failure() -> None:
    """验证 HTTP 200 但 retcode 非零时仍明确失败。

    Returns:
        None: 测试通过时无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    response = mock.MagicMock()
    response.__enter__.return_value.status = 200
    response.__enter__.return_value.read.return_value = json.dumps(
        {"status": "failed", "retcode": 1400, "message": "bad avatar"}
    ).encode("utf-8")

    with mock.patch.object(qq_group_bot, "urlopen", return_value=response):
        with pytest.raises(RuntimeError, match="retcode=1400"):
            qq_group_bot._call_onebot_action(
                "http://onebot",
                "set_qq_avatar",
                {"file": "base64://AAAA"},
            )
