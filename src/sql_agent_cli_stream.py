"""
基于 LangGraph + PostgresSaver 的交互式 Agent（流式 + 可暂停）。

改进点：
- LLM 消息改为流式输出，支持中途 Ctrl-C 暂停，确保已产生的内容不会丢失；
- 暂停后不再继续流式展示，但会用已获得内容完成后续图更新（如保存检查点）。

注意：
- 严禁硬编码密钥，环境变量读取：`OPENAI_API_KEY`、`TAVILY_API_KEY`、`LANGGRAPH_PG`；
- 必须在已激活的虚拟环境中运行；
- 不使用 argparse 参数解析；
- 测试可通过 `RUN_AGENT_TEST=1` 或 `DRY_RUN=1`（本地假模型模拟流式）运行。
"""

from __future__ import annotations

import os
import sys
import threading
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
# - OPENAI_API_KEY, TAVILY_API_KEY, LANGGRAPH_PG, THREAD_ID（可选）
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
        model_name (str): LLM 模型名称，默认 "openai:gpt-4o-mini"。
        pg_conn (str): Postgres 连接串，从 `LANGGRAPH_PG` 读取。
        thread_id (str): 会话线程 ID，默认 "demo-sql"。
        use_memory_ckpt (bool): 测试用内存检查点（DRY_RUN=1 时强制 True）。
    """

    model_name: str = "openai:gpt-4o-mini"
    pg_conn: str = ""
    thread_id: str = "demo-sql"
    use_memory_ckpt: bool = False
    enable_tools: bool = False


class SQLCheckpointAgentStreaming:
    """支持流式输出、可暂停、SQL 检查点历史与回放的交互式 Agent。

    改动要点：
    - 在 LLM 节点内部使用 `stream()` 逐块生成，并通过回调输出；
    - 监听“暂停事件”，一旦暂停，立刻以已产生内容构造消息，继续后续图更新；
    - 若用户 Ctrl-C 中断，本轮返回不为空的部分文本，并完成图更新。
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

        # DRY_RUN 时强制使用内存检查点，避免依赖外部 PG
        dry_run = os.environ.get("DRY_RUN") == "1"
        self._config = config
        if dry_run:
            self._config.use_memory_ckpt = True

        # 生产模式要求 PG 连接串；DRY_RUN/内存检查点可不要求
        if not self._config.use_memory_ckpt:
            assert (
                self._config.pg_conn
            ), "必须通过环境变量 LANGGRAPH_PG 提供 Postgres 连接串。"

        # 仅在需要真实模型时检查 OPENAI_API_KEY
        if self._config.model_name.startswith("openai:"):
            assert os.environ.get("OPENAI_API_KEY"), "缺少 OPENAI_API_KEY 环境变量。"

        self._saver: Optional[PostgresSaver | MemorySaver] = None
        # 工具开关：
        # - 若未显式设置 ENABLE_TOOLS，则当存在 TAVILY_API_KEY 时默认启用工具；
        # - 若显式设置 ENABLE_TOOLS，则以环境变量为准。
        env_tools = os.environ.get("ENABLE_TOOLS")
        if env_tools is None:
            self._enable_tools = bool(os.environ.get("TAVILY_API_KEY")) or config.enable_tools
        else:
            self._enable_tools = env_tools in {"1", "true", "True"}
        self._graph = self._build_graph()
        self._printed_in_round: bool = False
        self._resume_cfg: Optional[dict] = None
        self._agent_header_printed: bool = False
        self._agent_seq_for_round: int = 0

    # ------------------------- Graph 构建 -------------------------
    def _build_graph(self):
        """构建带工具与检查点（Postgres 或内存）的 LangGraph。

        Returns:
            Any: 已编译的图对象（LangGraph Graph）。

        Raises:
            RuntimeError: 当 Postgres 初始化失败时抛出。
        """
        # 初始化模型（可被 fake 覆盖）
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
                tavily_key = os.environ.get("TAVILY_API_KEY")
                tools = [TavilySearch(max_results=2)] if tavily_key else []
            # 首轮允许自动调用工具；二轮强制不再调用工具，产出最终答案
            llm_tools_auto = llm.bind_tools(tools) if tools else llm
            llm_tools_none = llm.bind_tools(tools, tool_choice="none") if tools else llm

        # 定义节点：支持流式，支持 Ctrl-C 暂停
        def chatbot(state: State):
            """核心对话节点：流式生成 + Ctrl-C 暂停。

            约定：
            - 将流式块通过 `on_token` 输出；
            - 若检测到暂停/中断，则用已收集内容构造最终消息返回；
            - 返回后，后续图（如检查点保存）继续照常执行。
            """
            # 输出回调：可被外部设置覆盖
            on_token: Callable[[str], None] = getattr(self, "_on_token", None) or (lambda s: None)
            partial: list[str] = []
            last_msg = None
            # 为模型提供系统级指令，确保使用工具后给出最终自然语言答案
            try:
                from langchain_core.messages import SystemMessage
                sys_msg = SystemMessage(
                    content=(
                        "You are a helpful Chinese assistant with web search tools.\n"
                        "When the user asks to 'search/find/look up/news/research' or needs fresh info, call the search tool first;\n"
                        "在工具返回后，用简洁中文总结关键信息并给出明确结论，必要时列要点。"
                    )
                )
                messages = [sys_msg] + list(state["messages"])  # 不修改原列表
            except Exception:
                messages = state["messages"]

            try:
                # 策略：
                # - 若当前状态尚未包含任何 Tool 消息：使用 invoke()，以确保工具调用元数据完整，便于 tools_condition 分支。
                # - 若已包含 Tool 消息（第二次进入 LLM 节点）：使用 stream() 流式输出最终自然语言结果。
                has_tool_feedback = False
                for m in state.get("messages", []):
                    t = getattr(m, "type", "") or getattr(m, "role", "")
                    if "tool" in str(t).lower():
                        has_tool_feedback = True
                        break

                if has_tool_feedback and hasattr(llm_tools_none, "stream"):
                    # 二轮：禁止再次调用工具，仅产出最终中文回答
                    # 追加提示，确保生成自然语言总结
                    from langchain_core.messages import SystemMessage
                    messages2 = [
                        SystemMessage(content="根据上文工具返回内容，用中文直接给出简洁清晰的最终回答，不要再调用工具。"),
                        *messages,
                    ]
                    for chunk in llm_tools_none.stream(messages2):
                        last_msg = chunk
                        text = getattr(chunk, "content", None)
                        if text:
                            partial.append(text)
                            on_token(text)
                    # 优先返回聚合文本，避免返回空 content 的尾块
                    from langchain_core.messages import AIMessage
                    aggregated = "".join(partial)
                    if aggregated:
                        return {"messages": [AIMessage(content=aggregated)]}
                    if last_msg is not None:
                        return {"messages": [last_msg]}
                    return {"messages": [AIMessage(content="")]} 

                # 首轮或无流能力：使用 invoke，确保工具调用结构保留
                # 首轮：允许自动调用工具
                msg = llm_tools_auto.invoke(messages)  # type: ignore[attr-defined]
                # 若直接得到文本（例如不需要工具），也输出给回调
                text = getattr(msg, "content", "")
                if isinstance(text, str) and text:
                    on_token(text)
                return {"messages": [msg]}

            except KeyboardInterrupt:
                # 中断：返回已产生内容，保证不为空（若确实为空也如实返回空字符串）
                from langchain_core.messages import AIMessage

                content = "".join(partial)
                return {"messages": [AIMessage(content=content)]}

        builder = StateGraph(State)
        builder.add_node("chatbot", chatbot)

        # 工具节点（若有）
        if tools:
            builder.add_node("tools", ToolNode(tools=tools))
            builder.add_edge(START, "chatbot")
            builder.add_conditional_edges("chatbot", tools_condition)
            builder.add_edge("tools", "chatbot")
        else:
            builder.add_edge(START, "chatbot")

        # 检查点：Postgres 或内存
        if self._config.use_memory_ckpt:
            self._saver = MemorySaver()
            graph = builder.compile(checkpointer=self._saver)
            return graph

        try:
            self._saver_cm = PostgresSaver.from_conn_string(self._config.pg_conn)
            self._saver = self._saver_cm.__enter__()
            self._saver.setup()
        except Exception as exc:
            raise RuntimeError(f"PostgresSaver 初始化失败：{exc}")

        graph = builder.compile(checkpointer=self._saver)
        return graph

    # ------------------------- 外部 API -------------------------
    @property
    def saver(self) -> PostgresSaver | MemorySaver:
        """返回内部检查点实例。"""
        assert self._saver is not None, "saver 尚未初始化。"
        return self._saver

    def set_token_printer(self, fn: Callable[[str], None]) -> None:
        """设置流式 token 输出回调：仅在接收到非空 token 时打印 Agent 行头。"""
        def _wrapped(s: str) -> None:
            # 忽略空白 token，避免出现“Agent: ”但无内容
            if s is None or (isinstance(s, str) and s.strip() == ""):
                return
            if not self._agent_header_printed and self._agent_seq_for_round:
                # 直接打印标题，不额外增加空行
                print("Agent: ", end="", flush=True)
                self._agent_header_printed = True
            self._printed_in_round = True
            fn(s)

        self._on_token = _wrapped

    def begin_round(self) -> None:
        """开始一轮对话前调用，用于重置状态。"""
        self._printed_in_round = False
        self._agent_header_printed = False
        self._agent_seq_for_round = 0

    def chat_once_stream(self, user_input: str, thread_id: Optional[str] = None) -> str:
        """执行一次对话（流式输出）。

        - 当用户按下 Ctrl-C 时，立即停止后续流式输出；
        - 函数返回截至中断时已产生的文本；
        - 后续图（如检查点保存）会基于“已获得文本”继续执行。

        Args:
            user_input (str): 用户输入。
            thread_id (str|None): 线程 ID。

        Returns:
            str: 本轮 LLM 已产生的文本（非空时不丢失）。
        """
        # 重置状态
        self.begin_round()

        # 若存在 time-travel 恢复点，先推进未完成的图（如待执行的 tools/chatbot）
        if self._resume_cfg is not None:
            try:
                st_resume = self._graph.get_state(self._resume_cfg)
                if st_resume and getattr(st_resume, "next", None) not in {(), ("__start__",)}:
                    # 先完成挂起节点，再进行新的用户输入，避免 tool_calls 语义错误
                    self._graph.invoke(None, self._resume_cfg)
            except Exception:
                pass
            # 清空恢复配置，回到常规行为
            self._resume_cfg = None

        # 基础配置：线程默认配置
        base_cfg = {"configurable": {"thread_id": thread_id or self._config.thread_id}}

        # 计算基准：当前线程上次终止检查点的消息数量（用于消息编号）
        base_tid = (base_cfg.get("configurable", {}) or {}).get("thread_id", self._config.thread_id)
        count_cfg = {"configurable": {"thread_id": base_tid}}
        prev_terms = self._terminal_states(count_cfg)
        prev_msgs = list(prev_terms[-1].values.get("messages", [])) if prev_terms else []
        base_msg_count = len(prev_msgs)
        # 仍保留“问答轮次”序号用于兼容
        seq = self._count_turns(count_cfg) + 1

        # 合并配置：在历史配置基础上仅添加/覆盖 seq
        cfg = dict(base_cfg)
        cfg.setdefault("configurable", {})
        cfg["configurable"] = dict(cfg["configurable"])  # 浅拷贝
        cfg["configurable"]["seq"] = seq
        last_text: str = ""
        tool_notified = False

        # 不重复打印用户消息（REPL 提示已包含 User: 前缀）
        agent_seq = base_msg_count + 2
        self._agent_seq_for_round = agent_seq

        try:
            # 使用图事件流，以捕获工具节点并在助手流式输出之前提示 Tool 调用
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
            # 捕获到中断，返回已得到的内容
            pass
        finally:
            # 若未产生流式 token，兜底打印 Agent 行（无序号）
            if not self._printed_in_round:
                if not last_text:
                    # 兜底：尝试从最终状态读取最后一条 AI 消息
                    try:
                        st = self._graph.get_state(cfg)
                        msgs = list(st.values.get("messages", [])) if st else []
                        # 从后往前找到第一条 AI 消息
                        for m in reversed(msgs):
                            if getattr(m, "type", "") == "ai":
                                txt = getattr(m, "content", "")
                                if isinstance(txt, str) and txt:
                                    last_text = txt
                                else:
                                    # 打印结构化内容的简要表示
                                    last_text = str(txt)
                                break
                    except Exception:
                        pass
                if last_text:
                    print(f"Agent: {last_text}")

            # 打印本轮新增的其他消息（例如 Tool），无序号
            try:
                st2 = self._graph.get_state(cfg)
                msgs2 = list(st2.values.get("messages", [])) if st2 else []
                delta = self._diff_messages(prev_msgs, msgs2)
                for m in delta:
                    label = self._role_label(m)
                    c = getattr(m, "content", "")
                    text = c if isinstance(c, str) else str(c)
                    if not isinstance(text, str) or text.strip() == "":
                        # 跳过空内容，避免出现“Agent: ”空行
                        continue
                    # 跳过用户行（已打印）
                    if label == "User":
                        continue
                    # 若 Agent 已经打印过最终行，避免重复打印空内容的中间 AI
                    if label == "Agent" and (self._printed_in_round or last_text):
                        continue
                    if label == "Tool":
                        # 运行时仅提示调用了哪个工具，不输出参数，避免噪音
                        name = getattr(m, "name", None) or "tool"
                        if not tool_notified:
                            print(f"Tool: Calling tool [{name}]")
                            tool_notified = True
                    else:
                        print(f"{label}: {text}")
            except Exception:
                pass
            # 若发生了流式输出，补一个换行，避免下一次提示接在同行
            if self._printed_in_round:
                print("")
        return last_text

    # ------------------------- 历史与序号辅助 -------------------------
    def _count_turns(self, cfg: dict) -> int:
        """统计某线程内已完成的问答轮数（以终止检查点计数）。

        Args:
            cfg (dict): graph 配置（包含 thread_id）。

        Returns:
            int: 已完成轮数。
        """
        states = list(self._graph.get_state_history(cfg))
        return sum(1 for st in states if getattr(st, "next", None) == ())

    def _terminal_states(self, cfg: dict):
        """按时间顺序返回终止检查点的状态列表（最早 -> 最新）。"""
        states = list(self._graph.get_state_history(cfg))
        terms = [st for st in states if getattr(st, "next", None) == ()]
        # get_state_history 通常是最近在前，这里反转为时间顺序
        return list(reversed(terms))

    @staticmethod
    def _diff_messages(prev: list, curr: list) -> list:
        """计算消息增量（假设使用 add_messages 追加）。"""
        if len(curr) >= len(prev) and curr[: len(prev)] == prev:
            return curr[len(prev) :]
        # 不满足前缀关系时，保守返回 curr 全量
        return curr

    @staticmethod
    def _role_label(m) -> str:
        """将消息类型映射为显示标签。"""
        t = getattr(m, "type", "") or getattr(m, "role", "")
        if t in {"human", "user"}:
            return "User"
        if t in {"ai", "assistant"}:
            return "Agent"
        if "tool" in str(t):
            return "Tool"
        return str(t or "Msg")

    @staticmethod
    def _extract_last_qa(messages: list) -> tuple[Optional[str], Optional[str]]:
        """从消息序列中抽取最后一个完整的问答（human -> ai）。"""
        last_ai: Optional[str] = None
        last_human: Optional[str] = None
        # 找到最后一条 AI
        for m in reversed(messages):
            if getattr(m, "type", "") == "ai":
                c = getattr(m, "content", "")
                last_ai = c if isinstance(c, str) else str(c)
                break
        # 在该 AI 之前找到最近一条 human
        if last_ai is not None:
            seen_ai = False
            for m in reversed(messages):
                t = getattr(m, "type", "")
                if t == "ai" and not seen_ai:
                    seen_ai = True
                    continue
                if seen_ai and t == "human":
                    c = getattr(m, "content", "")
                    last_human = c if isinstance(c, str) else str(c)
                    break
        return last_human, last_ai

    def list_history(self, thread_id: Optional[str] = None) -> list[str]:
        """按时间顺序列出检查点摘要（参考 timetravel）。

        输出每个检查点一行：
        - 形如：`[索引] node=<last_role> latest=<最后一条消息简述> messages=<总数> next=<后继节点>`
        - 索引从 0 开始，按时间顺序（最早 -> 最新）。

        Returns:
            list[str]: 每个检查点的摘要行。
        """
        cfg = {"configurable": {"thread_id": thread_id or self._config.thread_id}}
        states = list(self._graph.get_state_history(cfg))
        if not states:
            print("(历史为空)")
            return []
        # get_state_history 通常是最近在前，反转
        states = list(reversed(states))
        lines: list[str] = []
        prev_next = None
        for idx, st in enumerate(states):
            msgs: list = list(st.values.get("messages", []))
            # 计算“当前节点”= 上一个检查点的 next（首次视为 input）
            if prev_next is None:
                current_node = "input"
            else:
                # 可能是单个或多个后继，显示为逗号分隔
                if isinstance(prev_next, tuple):
                    if len(prev_next) == 0:
                        current_node = "terminal"
                    else:
                        current_node = ",".join(prev_next)
                else:
                    current_node = str(prev_next)

            # 计算“下一个节点”
            next_nodes = ",".join(st.next) if isinstance(st.next, tuple) else str(st.next)

            # 取最新可读文本（为空则回溯）
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
                # 仅对非 Tool 消息做截断，Tool 需要完整参数
                if last_role != "Tool" and isinstance(last_text, str) and len(last_text) > 120:
                    last_text = last_text[:117] + "..."

            line = f"[{idx}] node={current_node} latest={last_text} messages={len(msgs)} next={next_nodes}"
            print(line)
            lines.append(line)
            prev_next = st.next
        return lines

    def replay_from(self, index: int, thread_id: Optional[str] = None) -> None:
        """时间旅行到“检查点索引”对应的历史检查点（按时间顺序索引）。

        Args:
            index (int): 检查点索引（0..N-1），按时间顺序。
        """
        cfg = {"configurable": {"thread_id": thread_id or self._config.thread_id}}
        states = list(self._graph.get_state_history(cfg))
        assert states, "没有可用的检查点。"
        states = list(reversed(states))
        assert 0 <= index < len(states), f"索引越界：有效范围 0..{len(states)-1}"
        target = states[index]
        self._resume_cfg = dict(target.config)
        print(
            f"已切换到检查点 [{index}]，消息总数={len(target.values.get('messages', []))}。后续对话将从此状态继续。"
        )

    def close(self) -> None:
        """关闭底层持久化资源。"""
        if getattr(self, "_saver_cm", None) is not None:
            try:
                self._saver_cm.__exit__(None, None, None)
            finally:
                self._saver_cm = None


def _read_env_config() -> AgentConfig:
    """从环境变量读取配置并返回 AgentConfig。"""
    model = os.environ.get("MODEL_NAME", "openai:gpt-4o-mini")
    pg = os.environ.get("LANGGRAPH_PG", "")
    thread = os.environ.get("THREAD_ID", "demo-sql")
    return AgentConfig(model_name=model, pg_conn=pg, thread_id=thread)


def _print_help() -> None:
    """打印 REPL 内置命令帮助（按需显示）。"""
    print(
        "\n内置命令:\n"
        "  :help               显示帮助\n"
        "  :history            查看检查点历史（时间顺序）\n"
        "  :replay <index>     切换到指定检查点索引（时间顺序）\n"
        "  :thread <id>        切换当前线程 ID（不删除历史）\n"
        "  :newthread [id]     新建线程并切换（不删除历史）\n"
        "  :clear              创建新线程并切换（清空当前视图）\n"
        "  :exit               退出\n"
    )


def run_repl(agent: SQLCheckpointAgentStreaming) -> None:
    """启动交互式 REPL（流式 + 可暂停）。"""
    print(f"[REPL] thread={agent._config.thread_id}，输入 ':help' 查看命令，':exit' 退出。")

    # 默认 token 打印器：原样输出
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
            agent._resume_cfg = None
            print(f"已切换到线程：{agent._config.thread_id}")
            continue
        if text.startswith(":newthread") or text == ":clear":
            parts = text.split(maxsplit=1)
            import time
            new_id = (
                parts[1].strip() if len(parts) == 2 and parts[1].strip() else time.strftime("thread-%Y%m%d-%H%M%S")
            )
            agent._config.thread_id = new_id
            agent._resume_cfg = None
            print(f"已新建并切换到线程：{agent._config.thread_id}")
            continue

        # 单轮会话：支持 Ctrl-C 暂停
        agent.chat_once_stream(text)


def smoke_test() -> None:
    """最小冒烟测试：使用假模型（流式 echo），内存检查点。

    DRY_RUN=1 将启用：
    - 模型：fake:echo（模拟流式逐词输出）
    - 检查点：内存 MemorySaver
    """
    os.environ.setdefault("DRY_RUN", "1")
    cfg = _read_env_config()
    cfg.model_name = "fake:echo"
    cfg.use_memory_ckpt = True

    agent = SQLCheckpointAgentStreaming(cfg)
    agent.set_token_printer(lambda s: print(s, end="", flush=True))
    try:
        print("\n[SmokeTest] 发起一次对话（可手动 Ctrl-C 测试暂停）…")
        out = agent.chat_once_stream("请用 6 个词介绍 LangGraph 的特点。")
        print(f"\n[SmokeTest] 输出汇总：{out}")
        print("\n[SmokeTest] 检查点历史：")
        agent.list_history()
    finally:
        agent.close()


# ------------------------- 假模型：流式 Echo -------------------------
class _FakeStreamingEcho:
    """用于 DRY_RUN 的简单假模型：将最后一条 user 内容按词流式回显。

    仅用于本地自测，不访问网络。
    """

    def bind_tools(self, tools: list) -> "_FakeStreamingEcho":  # noqa: D401
        """保持接口一致，直接返回 self。"""
        return self

    def stream(self, messages: Iterable[dict]):
        """逐词流式输出最后一条 user 的内容。"""
        from langchain_core.messages import AIMessage

        last = None
        for m in messages:
            if isinstance(m, dict):
                if m.get("role") == "user":
                    last = m.get("content", "")
            else:
                # 兼容 Message 对象
                if getattr(m, "type", "") == "human":
                    last = getattr(m, "content", "")
        text = str(last or "")
        for token in text.split():
            time.sleep(0.05)
            yield AIMessage(content=token + " ")


if __name__ == "__main__":
    os.system("brew services start postgresql")
    # 当设置 RUN_AGENT_TEST=1 时，执行一次最小冒烟测试；否则进入 REPL
    if os.environ.get("RUN_AGENT_TEST") == "1":
        smoke_test()
    else:
        config = _read_env_config()
        agent = SQLCheckpointAgentStreaming(config)
        try:
            run_repl(agent)
        finally:
            agent.close()

    # 自动停止postgres服务（仅限 macOS + Homebrew）
    os.system("brew services stop postgresql")
    # 检查 PostgreSQL 是否已停止
    os.system("brew services list | grep postgresql")
