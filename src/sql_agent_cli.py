"""
基于 LangGraph + PostgresSaver 的交互式 Agent（安全版）。

特性：
- 使用 SQL（PostgreSQL）持久化检查点，支持历史查看与回放（时间旅行）。
- 全中文 Docstring 与显式类型标注。
- 无硬编码密钥：从环境变量读取 `OPENAI_API_KEY`、`TAVILY_API_KEY`、`LANGGRAPH_PG`。
- 遵循仓库规范：不使用 argparse 传参，提供交互式 REPL 内命令。

环境变量：
- OPENAI_API_KEY：OpenAI API Key（必需）
- TAVILY_API_KEY：Tavily API Key（可选，但启用工具时建议设置）
- LANGGRAPH_PG：Postgres 连接串，如 ***REMOVED***（必需）
- THREAD_ID：默认会话线程 ID（可选，默认 "demo-sql"）

使用示例：
1) 激活虚拟环境：
   source .venv/bin/activate
2) 设置环境变量（示例）：
   export OPENAI_API_KEY=sk-...
   export TAVILY_API_KEY=tvly-...
   export LANGGRAPH_PG=***REMOVED***
3) 运行：
   python sql_agent_cli.py
   - 在 REPL 中输入用户问题
   - 内置命令：
       :history           查看检查点历史
       :replay <index>    从指定历史检查点回放
       :exit              退出
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Annotated, Iterable, Optional

from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

"""严禁在代码中硬编码密钥，需从外部环境提供：
- OPENAI_API_KEY
- TAVILY_API_KEY
- LANGGRAPH_PG
"""


class State(TypedDict):
    """Agent 的图状态。

    Attributes:
        messages (list): 会话消息列表；使用 `add_messages` 作为 reducer 进行追加。
    """

    messages: Annotated[list, add_messages]


@dataclass
class AgentConfig:
    """Agent 运行配置。

    Attributes:
        model_name (str): LLM 模型名称，默认 "openai:gpt-4.1"。
        pg_conn (str): Postgres 连接串，从 `LANGGRAPH_PG` 读取。
        thread_id (str): 会话线程 ID，默认 "demo-sql"。
    """

    model_name: str = "openai:gpt-4.1"
    pg_conn: str = ""
    thread_id: str = "demo-sql"


class SQLCheckpointAgent:
    """支持 SQL 检查点历史与回放的交互式 Agent。

    该类负责：
    - 构建带工具节点的 LangGraph
    - 连接 PostgresSaver 以持久化检查点
    - 提供对话、历史查询与回放接口
    """

    def __init__(self, config: AgentConfig) -> None:
        """初始化 Agent。

        Args:
            config (AgentConfig): 运行配置。

        Raises:
            AssertionError: 当必要环境未满足（虚拟环境未激活、缺少连接串或 API Key）时抛出。
        """
        # 要求：必须使用虚拟环境
        assert os.environ.get("VIRTUAL_ENV") or sys.prefix.endswith(
            ".venv"
        ), "必须先激活虚拟环境 (.venv)。"

        # 要求：必须提供 Postgres 连接串
        assert config.pg_conn, "必须通过环境变量 LANGGRAPH_PG 提供 Postgres 连接串。"

        # 要求：OpenAI Key 必须存在（不在代码中硬编码）
        assert os.environ.get("OPENAI_API_KEY"), "缺少 OPENAI_API_KEY 环境变量。"

        self._config = config
        self._saver: Optional[PostgresSaver] = None
        self._graph = self._build_graph()

    def _build_graph(self):
        """构建带工具与 SQL 检查点的 LangGraph。

        Returns:
            Any: 已编译的图对象（LangGraph Graph）。

        Raises:
            RuntimeError: 当 Postgres 初始化失败时抛出。
        """
        # 初始化模型
        llm = init_chat_model(self._config.model_name)

        # 工具集（这里以 Tavily 搜索为例）
        tavily_key = os.environ.get("TAVILY_API_KEY")
        tools = [TavilySearch(max_results=2)] if tavily_key else []
        llm_with_tools = llm.bind_tools(tools) if tools else llm

        # 定义节点
        def chatbot(state: State):
            """核心对话节点：将状态消息输入 LLM 并返回回复。"""
            ai_msg = llm_with_tools.invoke(state["messages"])
            return {"messages": [ai_msg]}

        builder = StateGraph(State)
        builder.add_node("chatbot", chatbot)
        if tools:
            builder.add_node("tools", ToolNode(tools=tools))
            builder.add_edge(START, "chatbot")
            builder.add_conditional_edges("chatbot", tools_condition)
            builder.add_edge("tools", "chatbot")
        else:
            builder.add_edge(START, "chatbot")

        # 启用 PostgresSaver
        try:
            # from_conn_string 返回的是一个上下文管理器，需要显式进入以获取 saver 实例
            self._saver_cm = PostgresSaver.from_conn_string(self._config.pg_conn)
            self._saver = self._saver_cm.__enter__()
            self._saver.setup()
        except Exception as exc:  # 显式失败，不做回退
            raise RuntimeError(f"PostgresSaver 初始化失败：{exc}")

        graph = builder.compile(checkpointer=self._saver)
        return graph

    @property
    def saver(self) -> PostgresSaver:
        """返回内部 PostgresSaver 实例。"""
        assert self._saver is not None, "saver 尚未初始化。"
        return self._saver

    def chat_once(self, user_input: str, thread_id: Optional[str] = None) -> str:
        """执行一次对话并返回 Assistant 文本。

        Args:
            user_input (str): 用户输入文本。
            thread_id (str|None): 会话线程 ID；默认使用初始化配置。

        Returns:
            str: 助手最后一条文本回复。
        """
        cfg = {"configurable": {"thread_id": thread_id or self._config.thread_id}}
        last_text: str = ""
        for ev in self._graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            cfg,
            stream_mode="values",
        ):
            if "messages" in ev and ev["messages"]:
                msg = ev["messages"][-1]
                last_text = getattr(msg, "content", "")
                try:
                    msg.pretty_print()
                except Exception:
                    if last_text:
                        print(last_text)
        return last_text

    def list_history(self, thread_id: Optional[str] = None) -> list[str]:
        """列出某线程的检查点历史（最近在后），返回可读摘要列表。"""
        cfg = {"configurable": {"thread_id": thread_id or self._config.thread_id}}
        items: list[str] = []
        states = list(self._graph.get_state_history(cfg))
        if not states:
            print("(历史为空)")
            return items
        for idx, st in enumerate(states):
            num = len(st.values.get("messages", []))
            summary = f"[{idx}] messages={num} next={st.next}"
            print(summary)
            items.append(summary)
        return items

    def replay_from(self, index: int, thread_id: Optional[str] = None) -> None:
        """从检查点索引回放，打印事件输出。"""
        cfg = {"configurable": {"thread_id": thread_id or self._config.thread_id}}
        states = list(self._graph.get_state_history(cfg))
        assert states, "没有可用的检查点可回放。"
        assert 0 <= index < len(states), f"索引越界：有效范围 0..{len(states) - 1}"
        target = states[index]
        print(
            f"从检查点 #{index} 回放（messages={len(target.values.get('messages', []))}）"
        )
        for ev in self._graph.stream(None, target.config, stream_mode="values"):
            if "messages" in ev and ev["messages"]:
                msg = ev["messages"][-1]
                try:
                    msg.pretty_print()
                except Exception:
                    print(getattr(msg, "content", ""))

    def close(self) -> None:
        """关闭底层持久化资源。"""
        # 成对关闭上下文
        if getattr(self, "_saver_cm", None) is not None:
            try:
                self._saver_cm.__exit__(None, None, None)
            finally:
                self._saver_cm = None


def _read_env_config() -> AgentConfig:
    """从环境变量读取配置并返回 AgentConfig。"""
    model = os.environ.get("MODEL_NAME", "openai:gpt-4.1")
    pg = os.environ.get("LANGGRAPH_PG", "")
    thread = os.environ.get("THREAD_ID", "demo-sql")
    return AgentConfig(model_name=model, pg_conn=pg, thread_id=thread)


def _print_help() -> None:
    """打印 REPL 内置命令帮助。"""
    print(
        "\n内置命令:\n"
        "  :history            查看检查点历史\n"
        "  :replay <index>     从指定历史检查点回放\n"
        "  :exit               退出\n"
    )


def run_repl(agent: SQLCheckpointAgent) -> None:
    """启动交互式 REPL。

    REPL 说明：
    - 直接输入文本进行聊天；
    - 用冒号命令执行管理动作，如 `:history`、`:replay 1` 等。
    """
    print(
        f"[REPL] thread={agent._config.thread_id}，输入 ':history' 查看历史，':exit' 退出。"
    )

    while True:
        _print_help()
        try:
            text = input("User: ").strip()
        except KeyboardInterrupt:
            print("\n已退出。")
            break
        if not text:
            continue
        if text in {":exit", ":quit", ":q"}:
            print("\nAgent已关闭。")
            break
        if text == ":history":
            agent.list_history()
            continue
        if text.startswith(":replay"):
            parts = text.split()
            assert len(parts) == 2 and parts[1].isdigit(), "用法：:replay <index>"
            agent.replay_from(int(parts[1]))
            continue
        agent.chat_once(text)


def smoke_test() -> None:
    """最小冒烟测试：

    动作：
    1) 构建 Agent
    2) 执行一次对话
    3) 列出历史

    说明：
    - 不做任何“回退(fallback)”实现，若前提不满足将通过 assert 失败。
    """
    cfg = _read_env_config()
    agent = SQLCheckpointAgent(cfg)
    try:
        print("\n[SmokeTest] 发起一次对话…")
        agent.chat_once("用一句话介绍 LangGraph 是什么？")
        print("\n[SmokeTest] 检查点历史：")
        agent.list_history()
    finally:
        agent.close()


if __name__ == "__main__":
    # 当设置 RUN_AGENT_TEST=1 时，执行一次最小冒烟测试；否则进入 REPL
    if os.environ.get("RUN_AGENT_TEST") == "1":
        smoke_test()
    else:
        config = _read_env_config()
        agent = SQLCheckpointAgent(config)
        try:
            run_repl(agent)
        finally:
            agent.close()
