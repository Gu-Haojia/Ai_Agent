"""NapCat 账号资料与 Prompt 映射管理模块。"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class PromptAccountProfile:
    """表示单个 Prompt 对应的 NapCat 账号资料。

    Args:
        prompt_name (str): 不含扩展名的 Prompt 名称。
        nickname (str): 需要设置的 QQ 昵称。
        avatar_file (str): 配置中记录的头像文件名。
        avatar_base64 (str): 头像文件的纯 Base64 内容。

    Returns:
        None: 数据类初始化不返回额外值。

    Raises:
        None: 数据类本身不主动抛出异常。
    """

    prompt_name: str
    nickname: str
    avatar_file: str
    avatar_base64: str


class PromptAccountProfileManager:
    """读取并校验 Prompt 对应的 NapCat 账号资料。

    Args:
        config_path (Path): Prompt 账号资料 JSON 文件路径。
        avatar_dir (Path): 头像文件所在目录。

    Returns:
        None: 类初始化不返回额外值。

    Raises:
        AssertionError: 当初始化路径参数非法时抛出。
    """

    def __init__(self, config_path: Path, avatar_dir: Path) -> None:
        """初始化固定配置文件与头像目录。

        Args:
            config_path (Path): Prompt 账号资料 JSON 文件路径。
            avatar_dir (Path): 头像文件所在目录。

        Returns:
            None: 初始化不返回额外值。

        Raises:
            AssertionError: 当路径参数类型非法时抛出。
        """
        assert isinstance(config_path, Path), "config_path 必须为 Path"
        assert isinstance(avatar_dir, Path), "avatar_dir 必须为 Path"
        self._config_path = config_path.resolve()
        self._avatar_dir = avatar_dir.resolve()

    def resolve(self, prompt_name: str) -> Optional[PromptAccountProfile]:
        """查找 Prompt 对应资料，未登记时返回 ``None``。

        每次调用都会重新读取配置文件，使配置修改可在下一次 `/switch`
        时直接生效。

        Args:
            prompt_name (str): 不含扩展名的 Prompt 名称。

        Returns:
            Optional[PromptAccountProfile]: 已登记资料；未登记时返回 ``None``。

        Raises:
            AssertionError: 当配置文件、登记项或头像文件非法时抛出。
        """
        normalized_prompt_name = prompt_name.strip()
        assert normalized_prompt_name, "prompt_name 不能为空"
        assert self._config_path.is_file(), (
            f"账号资料配置文件不存在：{self._config_path}"
        )
        try:
            data = json.loads(self._config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"账号资料配置文件不是合法 JSON：{self._config_path}"
            ) from exc
        assert isinstance(data, dict), "账号资料配置文件根节点必须为对象"

        raw_profile = data.get(normalized_prompt_name)
        if raw_profile is None:
            return None
        assert isinstance(raw_profile, dict), (
            f"Prompt {normalized_prompt_name} 的账号资料必须为对象"
        )

        nickname = raw_profile.get("nickname")
        avatar_file = raw_profile.get("avatar_file")
        assert isinstance(nickname, str) and nickname.strip(), (
            f"Prompt {normalized_prompt_name} 的 nickname 不能为空"
        )
        assert isinstance(avatar_file, str) and avatar_file.strip(), (
            f"Prompt {normalized_prompt_name} 的 avatar_file 不能为空"
        )

        normalized_avatar_file = avatar_file.strip()
        assert "/" not in normalized_avatar_file, "avatar_file 只允许填写文件名"
        assert "\\" not in normalized_avatar_file, "avatar_file 只允许填写文件名"
        assert normalized_avatar_file not in {".", ".."}, (
            "avatar_file 不能使用相对目录"
        )

        avatar_path = (self._avatar_dir / normalized_avatar_file).resolve()
        assert avatar_path.parent == self._avatar_dir, (
            "avatar_file 必须位于 prompts/avatars 目录"
        )
        assert avatar_path.is_file(), f"头像文件不存在：{avatar_path}"
        avatar_bytes = avatar_path.read_bytes()
        assert avatar_bytes, f"头像文件不能为空：{avatar_path}"

        return PromptAccountProfile(
            prompt_name=normalized_prompt_name,
            nickname=nickname.strip(),
            avatar_file=normalized_avatar_file,
            avatar_base64=base64.b64encode(avatar_bytes).decode("ascii"),
        )
