"""
增强版：流式 Agent（多轮工具、先 Tool 后 Agent、强化综合总结）。

特性：
- 多轮工具：允许在工具结果后再次调用工具（由模型自行决定），直到不再需要工具为止；
- 输出顺序：先输出 Tool 调用行，再开始 Agent 流式内容；
- 强化系统提示：禁止生搬硬套搜索结果，要求围绕用户意图综合、提炼、给出可执行建议；
- REPL 仅在输入 :help 时显示帮助；
- 历史按时间顺序输出（timetravel 风格），支持索引回放。

"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Annotated, Callable, Iterable, Optional

from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# 说明：严禁在代码中硬编码密钥；请通过环境变量注入：

class State(TypedDict):
    """Agent 的图状态。"""

    messages: Annotated[list, add_messages]


@dataclass
class AgentConfig:
    """Agent 运行配置。"""

    model_name: str = "openai:gpt-4o-mini"
    pg_conn: str = ""
    thread_id: str = "demo-plus"
    use_memory_ckpt: bool = False
    enable_tools: bool = False


class SQLCheckpointAgentStreamingPlus:
    """多轮工具 + 强化综合 的流式 Agent。"""

    def __init__(self, config: AgentConfig) -> None:
        # 必须使用虚拟环境
        assert os.environ.get("VIRTUAL_ENV") or sys.prefix.endswith(
            ".venv"
        ), "必须先激活虚拟环境 (.venv)。"

        dry_run = os.environ.get("DRY_RUN") == "1"
        self._config = config
        if dry_run:
            self._config.use_memory_ckpt = True

        if not self._config.use_memory_ckpt:
            assert self._config.pg_conn, "必须通过 LANGGRAPH_PG 提供 Postgres 连接串。"
        if self._config.model_name.startswith("openai:"):
            assert os.environ.get("OPENAI_API_KEY"), "缺少 OPENAI_API_KEY 环境变量。"

        env_tools = os.environ.get("ENABLE_TOOLS")
        if env_tools is None:
            self._enable_tools = (
                bool(os.environ.get("TAVILY_API_KEY")) or config.enable_tools
            )
        else:
            self._enable_tools = env_tools in {"1", "true", "True"}

        # 预先读取并缓存系统提示内容（从外部文件），避免每轮重复IO
        self._sys_msg_content: str = self._load_sys_msg_content()

        self._graph = self._build_graph()
        self._printed_in_round: bool = False

    def _load_sys_msg_content(self) -> str:
        """读取系统提示词内容。

        优先从环境变量 `SYS_MSG_FILE` 指定的路径读取系统提示文本；
        要求该文件必须存在且非空，否则抛出断言错误。

        Returns:
            str: 系统提示词全文内容。

        Raises:
            AssertionError: 当环境变量未设置、文件不存在或内容为空时抛出。
        """
        path = os.environ.get("SYS_MSG_FILE")
        assert (
            path
        ), "必须通过环境变量 SYS_MSG_FILE 指定系统提示文件路径。"
        abs_path = os.path.abspath(path)
        assert os.path.isfile(abs_path), f"系统提示文件不存在: {abs_path}"
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()
            print(f"[系统提示] 已加载文件: {abs_path}，长度 {len(content)} 字符。")
            #打印头尾各50字符,仅输出文本不要格式符号
            print(f"[系统提示] Prompt内容预览: {content[:50].replace(chr(10), ' ')} ... {content[-50:].replace(chr(10), ' ')}")
        assert content and content.strip(), "系统提示文件内容为空。"
        return content

    def _build_graph(self):
        model_name = self._config.model_name
        if model_name == "fake:echo":
            llm = _FakeStreamingEcho()
            tools: list = []
            llm_tools_auto = llm
            llm_tools_none = llm
        else:
            llm = init_chat_model(model_name)
            tools = []
            if self._enable_tools:
                from langchain_tavily import TavilySearch

                if os.environ.get("TAVILY_API_KEY"):
                    tools = [TavilySearch(max_results=3)]

            # 两种绑定：
            # - auto：允许模型自行决定是否调用工具（用于首轮/工具前）
            # - none：禁止工具，促使模型基于工具结果做总结（用于工具后）
            llm_tools_auto = llm.bind_tools(tools) if tools else llm
            llm_tools_none = llm.bind_tools(tools, tool_choice="none") if tools else llm
            # - force：当用户显式要求搜索/检索等，强制调用 tavily_search
            if tools:
                llm_tools_force = llm.bind_tools(
                    tools,
                    tool_choice={
                        "type": "function",
                        "function": {"name": "tavily_search"},
                    },
                )
            else:
                llm_tools_force = llm

        def chatbot(state: State):
            on_token: Callable[[str], None] = getattr(self, "_on_token", None) or (
                lambda s: None
            )
            partial: list[str] = []
            last_msg = None
            # 系统提示：使用初始化时缓存的外部文件内容，综合 & 不生搬硬套搜索结果
            try:
                from langchain_core.messages import SystemMessage
                sys_msg = SystemMessage(content=self._sys_msg_content)
                messages = [sys_msg] + list(state["messages"])  # 不修改原列表
            except Exception:
                messages = state["messages"]

            """
            # 策略增强：
            # - 仅统计“最后一条 human 之后”的 Tool 消息，避免跨轮次误判。
            # - 显式搜索请求或“继续”且上次 AI 承诺搜索时，强制调用工具。
            msgs = list(state.get("messages", []))
            last_human_idx = -1
            for i in range(len(msgs) - 1, -1, -1):
                if (
                    getattr(msgs[i], "type", "") == "human"
                    or getattr(msgs[i], "role", "") == "user"
                ):
                    last_human_idx = i
                    break
            has_tool_feedback = any(
                (
                    "tool" in str(getattr(m, "type", "")).lower()
                    or "tool" in str(getattr(m, "role", "")).lower()
                )
                for m in msgs[last_human_idx + 1 :]
            )

            def _needs_search(text: str) -> bool:
                if not isinstance(text, str):
                    return False
                zh = [
                    "搜索",
                    "检索",
                    "查找",
                    "互联网",
                    "网页",
                    "上网",
                    "先用工具",
                    "帮我搜",
                ]
                en = [
                    "search",
                    "find",
                    "look up",
                    "lookup",
                    "news",
                    "research",
                    "investigate",
                ]
                t = text.lower()
                return any(k in text for k in zh) or any(k in t for k in en)

            def _last_ai_promised_search() -> bool:
                for m in reversed(msgs[:last_human_idx]):
                    if getattr(m, "type", "") == "ai":
                        c = getattr(m, "content", "")
                        if not isinstance(c, str):
                            c = str(c)
                        c_low = c.lower()
                        if any(
                            x in c
                            for x in [
                                "搜索",
                                "检索",
                                "查找",
                                "我将为您搜索",
                                "我会搜索",
                                "请稍等",
                            ]
                        ):
                            return True
                        if any(
                            x in c_low
                            for x in [
                                "search",
                                "i will search",
                                "i'll search",
                                "looking up",
                                "please wait",
                            ]
                        ):
                            return True
                        break
                return False

            last_user_text = ""
            if last_human_idx >= 0:
                c = getattr(msgs[last_human_idx], "content", "")
                last_user_text = c if isinstance(c, str) else str(c)
            should_force_tool = _needs_search(last_user_text) or (
                last_user_text.strip() in {"继续", "go on", "continue"}
                and _last_ai_promised_search()
            )
            """
            """
            if has_tool_feedback and hasattr(llm_tools_none, "stream"):
                for chunk in llm_tools_none.stream(messages):
                    last_msg = chunk
                    txt = getattr(chunk, "content", None)
                    if txt:
                        partial.append(txt)
                        on_token(txt)
                from langchain_core.messages import AIMessage

                aggregated = "".join(partial)
                if aggregated:
                    return {"messages": [AIMessage(content=aggregated)]}
                if last_msg is not None:
                    return {"messages": [last_msg]}
                return {"messages": [AIMessage(content="")]}  # 空聚合兜底
            """

            # 首轮/无工具反馈：同样改为流式输出
            # 显式要求则强制工具，否则交由模型自动决定
            runner = llm_tools_auto

            if hasattr(runner, "stream"):
                # 使用 LangChain 的 chunk 相加协议，将增量内容与工具调用一起合并
                accumulated = None
                for c in runner.stream(messages):  # type: ignore[attr-defined]
                    txt = getattr(c, "content", None)
                    if txt:
                        on_token(txt)
                    accumulated = c if accumulated is None else accumulated + c

                if accumulated is not None:
                    return {"messages": [accumulated]}

            # 退化路径：不支持流式则一次性返回
            msg = runner.invoke(messages)  # type: ignore[attr-defined]
            txt = getattr(msg, "content", "")
            if isinstance(txt, str) and txt:
                on_token(txt)
            return {"messages": [msg]}

        builder = StateGraph(State)
        builder.add_node("chatbot", chatbot)

        if tools:
            builder.add_node("tools", ToolNode(tools=tools))
            builder.add_edge(START, "chatbot")
            builder.add_conditional_edges("chatbot", tools_condition)
            builder.add_edge("tools", "chatbot")
        else:
            builder.add_edge(START, "chatbot")

        if self._config.use_memory_ckpt:
            self._saver = MemorySaver()
            return builder.compile(checkpointer=self._saver)

        try:
            self._saver_cm = PostgresSaver.from_conn_string(self._config.pg_conn)
            self._saver = self._saver_cm.__enter__()
            self._saver.setup()
        except Exception as exc:
            raise RuntimeError(f"PostgresSaver 初始化失败：{exc}")

        return builder.compile(checkpointer=self._saver)

    # --------------- 外部 API ---------------
    def set_token_printer(self, fn: Callable[[str], None]) -> None:
        def _wrapped(s: str) -> None:
            if s is None or (isinstance(s, str) and s.strip() == ""):
                return
            if not getattr(self, "_agent_header_printed", False):
                print("Agent: ", end="", flush=True)
                self._agent_header_printed = True
            self._printed_in_round = True
            fn(s)

        self._on_token = _wrapped

    def chat_once_stream(self, user_input: str, thread_id: Optional[str] = None) -> str:
        # 每轮初始化
        self._printed_in_round = False
        self._agent_header_printed = False
        cfg = {"configurable": {"thread_id": thread_id or self._config.thread_id}}
        last_text = ""
        tool_notified = False

        try:
            for ev in self._graph.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                cfg,
                stream_mode="values",
            ):
                if not (isinstance(ev, dict) and "messages" in ev and ev["messages"]):
                    continue
                m = ev["messages"][-1]
                label = self._role_label(m)
                if label == "Tool" and not tool_notified:
                    name = getattr(m, "name", None) or "tool"
                    print(f"Tool: Calling tool [{name}]")
                    tool_notified = True
                if label == "Agent":
                    txt = getattr(m, "content", "")
                    if isinstance(txt, str) and txt:
                        last_text = txt
        except KeyboardInterrupt:
            print("\n暂停生成。")
            pass
        finally:
            if not self._printed_in_round and last_text:
                print(f"Agent: {last_text}")
            if self._printed_in_round:
                print("")
        return last_text

    # --------------- 历史/回放 ---------------
    @staticmethod
    def _role_label(m) -> str:
        t = getattr(m, "type", "") or getattr(m, "role", "")
        if t in {"human", "user"}:
            return "User"
        if t in {"ai", "assistant"}:
            return "Agent"
        if "tool" in str(t):
            return "Tool"
        return str(t or "Msg")

    def list_history(self, thread_id: Optional[str] = None) -> list[str]:
        cfg = {"configurable": {"thread_id": thread_id or self._config.thread_id}}
        states = list(self._graph.get_state_history(cfg))
        if not states:
            print("(历史为空)")
            return []
        states = list(reversed(states))
        lines: list[str] = []
        prev_next = None
        for idx, st in enumerate(states):
            msgs: list = list(st.values.get("messages", []))
            # 当前节点：上个检查点的 next
            if prev_next is None:
                current_node = "input"
            else:
                if isinstance(prev_next, tuple):
                    current_node = (
                        "terminal" if len(prev_next) == 0 else ",".join(prev_next)
                    )
                else:
                    current_node = str(prev_next)
            next_nodes = (
                ",".join(st.next) if isinstance(st.next, tuple) else str(st.next)
            )

            # 最新可读文本（Tool 不截断）
            last_text = ""
            last_role = None
            if msgs:
                m = msgs[-1]
                last_role = self._role_label(m)
                c = getattr(m, "content", "")
                last_text = c if isinstance(c, str) else str(c)
                if not (isinstance(last_text, str) and last_text.strip()):
                    for mm in reversed(msgs):
                        last_role = self._role_label(mm)
                        tt = getattr(mm, "content", "")
                        tt = tt if isinstance(tt, str) else str(tt)
                        if isinstance(tt, str) and tt.strip():
                            last_text = tt
                            break
                if (
                    last_role != "Tool"
                    and isinstance(last_text, str)
                    and len(last_text) > 120
                ):
                    last_text = last_text[:117] + "..."

            line = f"[{idx}] node={current_node} latest={last_text} messages={len(msgs)} next={next_nodes}"
            print(line)
            lines.append(line)
            prev_next = st.next
        return lines

    def replay_from(self, index: int, thread_id: Optional[str] = None) -> None:
        cfg = {"configurable": {"thread_id": thread_id or self._config.thread_id}}
        states = list(self._graph.get_state_history(cfg))
        assert states, "没有可用的检查点。"
        states = list(reversed(states))
        assert 0 <= index < len(states), f"索引越界：0..{len(states)-1}"
        target = states[index]
        print(f"从检查点 [{index}] 回放 …")
        for ev in self._graph.stream(None, target.config, stream_mode="values"):
            if "messages" in ev and ev["messages"]:
                m = ev["messages"][-1]
                try:
                    m.pretty_print()
                except Exception:
                    print(getattr(m, "content", ""))


def _read_env_config() -> AgentConfig:
    model = os.environ.get("MODEL_NAME", "openai:gpt-4o-mini")
    pg = os.environ.get("LANGGRAPH_PG", "")
    thread = os.environ.get("THREAD_ID", "demo-plus")
    return AgentConfig(model_name=model, pg_conn=pg, thread_id=thread)


def _print_help() -> None:
    print(
        "\n内置命令:\n"
        "  :help               显示帮助\n"
        "  :history            查看检查点历史（时间顺序）\n"
        "  :replay <index>     从指定检查点索引回放\n"
        "  :thread <id>        切换当前线程 ID\n"
        "  :newthread [id]     新建线程并切换\n"
        "  :clear              创建新线程并切换\n"
        "  :exit               退出\n"
    )


def run_repl(agent: SQLCheckpointAgentStreamingPlus) -> None:
    print(
        f"[REPL] thread={agent._config.thread_id}，输入 ':help' 查看命令，':exit' 退出。"
    )
    agent.set_token_printer(lambda s: print(s, end="", flush=True))
    while True:
        try:
            text = input("User: ").strip()
        except KeyboardInterrupt:
            print("\n已退出。")
            break
        except EOFError:
            print("\n输入流结束，已退出。")
            break
        if not text:
            continue
        if text in {":exit", ":quit", ":q"}:
            print("\nAgent已关闭。")
            break
        if text == ":help":
            _print_help()
            continue
        if text == ":history":
            agent.list_history()
            continue
        if text.startswith(":replay"):
            parts = text.split()
            assert len(parts) == 2 and parts[1].isdigit(), "用法：:replay <index>"
            agent.replay_from(int(parts[1]))
            continue
        if text.startswith(":thread"):
            parts = text.split(maxsplit=1)
            assert len(parts) == 2 and parts[1], "用法：:thread <id>"
            agent._config.thread_id = parts[1].strip()
            print(f"已切换到线程：{agent._config.thread_id}")
            continue
        if text.startswith(":newthread") or text == ":clear":
            parts = text.split(maxsplit=1)
            new_id = (
                parts[1].strip()
                if len(parts) == 2 and parts[1].strip()
                else time.strftime("thread-%Y%m%d-%H%M%S")
            )
            agent._config.thread_id = new_id
            print(f"已新建并切换到线程：{agent._config.thread_id}")
            continue

        agent.chat_once_stream(text)


# ------------------------- 假模型：流式 Echo -------------------------
class _FakeStreamingEcho:
    def bind_tools(self, tools: list) -> "_FakeStreamingEcho":
        return self

    def stream(self, messages: Iterable[dict]):
        from langchain_core.messages import AIMessage

        last = None
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "user":
                last = m.get("content", "")
        text = str(last or "")
        for token in text.split():
            time.sleep(0.05)
            yield AIMessage(content=token + " ")


if __name__ == "__main__":
    # 按需启动/停止本机 brew postgresql，便于你的调试
    os.system("brew services start postgresql")
    # wait for postgresql to be ready
    while True:
        res = os.system("pg_isready -q")
        if res == 0:
            break
        time.sleep(1)

    if os.environ.get("RUN_AGENT_TEST") == "1":
        cfg = _read_env_config()
        cfg.model_name = "fake:echo"
        cfg.use_memory_ckpt = True
        agent = SQLCheckpointAgentStreamingPlus(cfg)
        agent.set_token_printer(lambda s: print(s, end="", flush=True))
        agent.chat_once_stream("测试 echo 模型 是否按词流式输出")
    else:
        config = _read_env_config()
        agent = SQLCheckpointAgentStreamingPlus(config)
        try:
            run_repl(agent)
        finally:
            pass

    os.system("brew services stop postgresql")
    os.system("brew services list | grep postgresql")
