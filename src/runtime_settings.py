"""管理 QQ Bot 可在运行时修改的持久化设置。"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


RUNTIME_SETTINGS_PATH: Path = Path(".runtime_settings.json")
MIN_TAVILY_SEARCH_LIMIT: int = 5
MAX_TAVILY_SEARCH_LIMIT: int = 999


@dataclass(frozen=True)
class RuntimeSettings:
    """保存可在运行时修改的 Agent 设置。

    Args:
        schema_version (int): 配置文件结构版本。
        tavily_search_limit (int): 单轮 Tavily 搜索提醒阈值。
        prompt_file (str): 持久化的 Prompt 文件名，空字符串表示使用环境变量。

    Returns:
        None: dataclass 初始化不返回额外值。

    Raises:
        AssertionError: 当版本或搜索上限不合法时抛出。
    """

    schema_version: int = 1
    tavily_search_limit: int = 5
    prompt_file: str = ""

    def __post_init__(self) -> None:
        """校验运行时设置。

        Returns:
            None: 校验通过后不返回额外值。

        Raises:
            AssertionError: 当版本或搜索上限不合法时抛出。
        """
        assert type(self.schema_version) is int, "schema_version 必须为整数"
        assert self.schema_version == 1, "schema_version 必须为 1"
        assert type(self.tavily_search_limit) is int, (
            "tavily_search_limit 必须为整数"
        )
        assert (
            MIN_TAVILY_SEARCH_LIMIT
            <= self.tavily_search_limit
            <= MAX_TAVILY_SEARCH_LIMIT
        ), "tavily_search_limit 必须在 5 到 999 之间"
        assert isinstance(self.prompt_file, str), "prompt_file 必须为字符串"


class RuntimeSettingsStore:
    """从固定 JSON 文件加载并保存运行时设置。

    Args:
        path (Path): 配置文件路径，默认使用项目目录下的固定文件名。

    Returns:
        None: 类初始化不返回额外值。

    Raises:
        AssertionError: 当路径不是文件路径时抛出。
    """

    def __init__(self, path: Path = RUNTIME_SETTINGS_PATH) -> None:
        """初始化运行时设置存储。

        Args:
            path (Path): 配置文件路径。

        Raises:
            AssertionError: 当路径不是文件路径时抛出。
        """
        assert isinstance(path, Path), "path 必须为 Path"
        assert path.name, "path 必须包含文件名"
        self._path = path

    def load(self) -> RuntimeSettings:
        """加载设置，文件不存在时创建默认配置。

        Returns:
            RuntimeSettings: 已校验的运行时设置。

        Raises:
            AssertionError: 当 JSON 结构或字段不合法时抛出。
            json.JSONDecodeError: 当配置文件不是合法 JSON 时抛出。
            OSError: 当配置文件无法读取或写入时抛出。
        """
        if not self._path.exists():
            settings = RuntimeSettings()
            self.save(settings)
            return settings
        data = json.loads(self._path.read_text(encoding="utf-8"))
        assert isinstance(data, dict), "运行时设置必须是 JSON 对象"
        assert set(data) == {
            "schema_version",
            "tavily_search_limit",
            "prompt_file",
        }, (
            "运行时设置字段不完整或包含未知字段"
        )
        return RuntimeSettings(
            schema_version=data["schema_version"],
            tavily_search_limit=data["tavily_search_limit"],
            prompt_file=data["prompt_file"],
        )

    def save(self, settings: RuntimeSettings) -> None:
        """原子保存运行时设置。

        Args:
            settings (RuntimeSettings): 待保存的运行时设置。

        Returns:
            None: 保存完成后不返回额外值。

        Raises:
            AssertionError: 当设置类型不正确时抛出。
            OSError: 当配置文件无法写入时抛出。
        """
        assert isinstance(settings, RuntimeSettings), "settings 类型非法"
        temporary_path = self._path.with_name(self._path.name + ".tmp")
        temporary_path.write_text(
            json.dumps(asdict(settings), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, self._path)
