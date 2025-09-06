from typing import Annotated

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

import os

# 说明：密钥请通过环境变量注入（OPENAI_API_KEY、TAVILY_API_KEY）


# 状态定义
class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]


# 定义搜索工具
tool = TavilySearch(max_results=2)
tools = [tool, human_assistance]

# 将 tools 绑定到 LLM 上
llm = init_chat_model("openai:gpt-4.1")
llm_with_tools = llm.bind_tools(tools)


# 定义主要节点
def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}


graph_builder.add_node("chatbot", chatbot)

# 新增 ToolNode 节点
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# 配置条件边：判断是否调用工具
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# 任意时工具调用完都返回 chatbot 继续
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)


# 运行（简易 REPL）
def stream_graph_updates(user_input: str, config):
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        # print(event)
        if "messages" in event:
            event["messages"][-1].pretty_print()

    # 检查是否中断
    snapshot = graph.get_state(config)
    if snapshot.interrupts:
        print("\n[!] interrupted, waiting for input...")
        human_input = input("Human> ")
        cmd = Command(resume={"data": human_input})
        events = graph.stream(cmd, config, stream_mode="values")
        for ev in events:
            # print(ev)
            if "messages" in ev:
                ev["messages"][-1].pretty_print()


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input, config)
        except KeyboardInterrupt:
            break
