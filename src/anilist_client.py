"""
封装 AniList GraphQL API 的精简客户端。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

ANILIST_MEDIA_SORTS: list[str] = [
    "TRENDING_DESC",
    "POPULARITY_DESC",
    "SCORE_DESC",
]
"""AniList 中最常用的 MediaSort 排序枚举。"""

_VALID_SEASONS: set[str] = {"WINTER", "SPRING", "SUMMER", "FALL"}


@dataclass
class AniListAPI:
    """
    AniList 图形接口客户端，提供基础的作品检索能力。

    Args:
        endpoint (str): GraphQL 接口地址。
        timeout (int): 网络请求超时时间，单位秒。
    """

    endpoint: str = "https://graphql.anilist.co"
    timeout: int = 15

    def __post_init__(self) -> None:
        """
        校验初始化参数。

        Raises:
            AssertionError: 当接口地址或超时配置非法时抛出。
        """

        assert isinstance(self.endpoint, str) and self.endpoint.startswith(
            "http"
        ), "endpoint 必须为合法 URL"
        assert isinstance(self.timeout, int) and self.timeout > 0, "timeout 必须为正整数"

    def search_media(
        self,
        query: str,
        season_year: int | None = None,
        season: str | None = None,
        sort: str | None = None,
        per_page: int = 6,
    ) -> dict[str, Any]:
        """
        按关键词检索 AniList 作品，并保留原始返回结构。

        Args:
            query (str): 搜索关键词；当 ``season`` 或 ``season_year`` 至少指定一项时可为空字符串。
            season_year (int | None): 过滤季节年份，范围 1900-2100。
            season (str | None): 过滤季度，可选 winter/spring/summer/fall。
            sort (str | None): 排序枚举，详见 ``ANILIST_MEDIA_SORTS``。
            per_page (int): 每次返回的条目数量，默认 6。

        Returns:
            dict[str, Any]: 包含 ``pageInfo`` 与 ``media`` 列表的原始数据。

        Raises:
            AssertionError: 当参数非法时抛出。
            ValueError: 当 API 调用失败或返回结构异常时抛出。
        """

        assert isinstance(query, str), "query 必须为字符串"
        assert isinstance(per_page, int) and 1 <= per_page <= 10, "per_page 必须介于 1-10"
        if season_year is not None:
            assert isinstance(season_year, int) and 1900 <= season_year <= 2100, (
                "season_year 必须介于 1900-2100"
            )

        normalized_season = self._normalize_season(season)
        normalized_sort = self._normalize_sort(sort)
        search_value = query.strip()
        if not search_value:
            assert (
                season_year is not None or normalized_season is not None
            ), "query 为空时必须提供 season 或 season_year 作为过滤条件"

        query_text = """
        query (
            $search: String!,
            $page: Int!,
            $perPage: Int!,
            $season: MediaSeason,
            $seasonYear: Int,
            $sort: [MediaSort!]
        ) {
            Page(page: $page, perPage: $perPage) {
                pageInfo {
                    total
                    perPage
                    currentPage
                    lastPage
                    hasNextPage
                }
                media(
                    search: $search,
                    season: $season,
                    seasonYear: $seasonYear,
                    sort: $sort
                ) {
                    id
                    title {
                        romaji
                        native
                        english
                    }
                    type
                    format
                    status
                    season
                    seasonYear
                    episodes
                    chapters
                    duration
                    averageScore
                    meanScore
                    popularity
                    favourites
                    description(asHtml: true)
                    genres
                    tags {
                        name
                        rank
                        isMediaSpoiler
                    }
                    coverImage {
                        extraLarge
                        large
                        medium
                        color
                    }
                    bannerImage
                    trailer {
                        id
                        site
                        thumbnail
                    }
                    siteUrl
                    startDate {
                        year
                        month
                        day
                    }
                    endDate {
                        year
                        month
                        day
                    }
                    studios(isMain: true) {
                        edges {
                            node {
                                name
                            }
                        }
                    }
                }
            }
        }
        """

        data = self._post(
            query_text,
            {
                "search": search_value,
                "page": 1,
                "perPage": per_page,
                "season": normalized_season,
                "seasonYear": season_year,
                "sort": normalized_sort,
            },
        )

        page = data.get("Page")
        if not isinstance(page, dict):
            raise ValueError("AniList 返回数据缺少 Page 节点")
        page_info = page.get("pageInfo")
        media_items = page.get("media")
        if not isinstance(page_info, dict) or not isinstance(media_items, list):
            raise ValueError("AniList 返回数据缺少 pageInfo 或 media 列表")
        return {"pageInfo": page_info, "media": media_items}

    def _post(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        """
        执行 GraphQL POST 请求。

        Args:
            query (str): GraphQL 查询语句。
            variables (dict[str, Any]): 查询参数。

        Returns:
            dict[str, Any]: data 节点。

        Raises:
            AssertionError: 当 query 为空时抛出。
            ValueError: 当请求失败或返回异常结构时抛出。
        """

        assert isinstance(query, str) and query.strip(), "GraphQL 查询语句不能为空"
        response = requests.post(
            self.endpoint,
            json={"query": query, "variables": variables},
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=self.timeout,
        )

        if response.status_code != 200:
            raise ValueError(f"AniList 请求失败，HTTP {response.status_code}")
        try:
            payload = response.json()
        except Exception as exc:
            raise ValueError("AniList 返回非 JSON 数据") from exc

        errors = payload.get("errors")
        if errors:
            first_error = errors[0] if isinstance(errors, list) and errors else errors
            if isinstance(first_error, dict):
                message = first_error.get("message") or str(first_error)
            else:
                message = str(first_error)
            raise ValueError(f"AniList 返回错误: {message}")

        data = payload.get("data")
        if not isinstance(data, dict):
            raise ValueError("AniList 返回数据缺失 data 节点")
        return data

    def _normalize_season(self, value: str | None) -> str | None:
        """
        规范化季度字符串。

        Args:
            value (str | None): 原始季度。

        Returns:
            str | None: AniList 支持的季度枚举。

        Raises:
            AssertionError: 当季度值非法时抛出。
        """

        if value is None:
            return None
        normalized = value.strip().upper()
        if normalized == "AUTUMN":
            normalized = "FALL"
        assert normalized in _VALID_SEASONS, "season 仅支持 winter/spring/summer/fall"
        return normalized

    def _normalize_sort(self, value: str | None) -> list[str]:
        """
        规范化排序参数。

        Args:
            value (str | None): 排序枚举。

        Returns:
            list[str]: GraphQL 接受的排序列表。

        Raises:
            AssertionError: 当排序值不在支持范围时抛出。
        """

        if value is None:
            return ["TRENDING_DESC"]
        normalized = value.strip().upper().replace("-", "_")
        assert normalized in ANILIST_MEDIA_SORTS, "sort 不在 ANILIST_MEDIA_SORTS 列表内"
        return [normalized]
