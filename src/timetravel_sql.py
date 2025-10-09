from typing import Annotated
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# 1) 初始化 LLM（用 OpenAI 举例，你也可以换 Claude、Gemini、Bedrock）
import os

# 说明：密钥请通过环境变量注入（OPENAI_API_KEY、TAVILY_API_KEY）

llm = init_chat_model("openai:gpt-4.1")


# 2) 定义 State
class State(TypedDict):
    messages: Annotated[list, add_messages]  # 消息列表，合并策略=追加


# 3) 定义节点
tool = TavilySearch(max_results=2)  # 一个搜索工具
tools = [tool]
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    # 读取 state["messages"]，用 LLM 推理
    ai_msg = llm_with_tools.invoke(state["messages"])
    return {"messages": [ai_msg]}


# 4) 搭建图
builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools=tools))

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")

# 5) 启用检查点（PostgreSQL 持久化）
conn_str = os.getenv(
    "LANGGRAPH_PG", "***REMOVED***"
)
with PostgresSaver.from_conn_string(conn_str) as memory:
    memory.setup()
    graph = builder.compile(checkpointer=memory)

    # ------------------ 执行一次对话 ------------------
    config = {"configurable": {"thread_id": "demo-3"}}
    print(">>> 执行一次完整对话（产生检查点）")
    events = graph.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "I'm learning LangGraph. Could you research it?",
                }
            ]
        },
        config,
        stream_mode="values",
    )
    for ev in events:
        if "messages" in ev:
            ev["messages"][-1].pretty_print()


    events = graph.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Ya that's helpful. Please "
                        "figure out how to build an autonomous agent with it!"
                    ),
                },
            ],
        },
        config,
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()

    # ------------------ 查看检查点历史 ------------------
    print("\n>>> 检查点历史")
    to_replay = None
    for state in graph.get_state_history(config):
        print("Num Messages:", len(state.values["messages"]), "Next:", state.next)
        print("-" * 40)
        if len(state.values["messages"]) == 6:  # 找个中间点
            to_replay = state

    # ------------------ 时间旅行：从历史继续 ------------------
    print("\n>>> 从检查点继续（时间旅行）")
    for ev in graph.stream(None, to_replay.config, stream_mode="values"):
        if "messages" in ev:
            ev["messages"][-1].pretty_print()

    # ------------------ 再次查看检查点历史 ------------------
    print("\n>>> 回溯后检查点历史")
    to_replay = None
    for state in graph.get_state_history(config):
        print("Num Messages:", len(state.values["messages"]), "Next:", state.next)
        print("-" * 40)
