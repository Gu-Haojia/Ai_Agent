from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
import os
# 说明：请通过环境变量提供 OPENAI_API_KEY、TAVILY_API_KEY


# 状态定义
class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

# 定义搜索工具
tool = TavilySearch(max_results=2)
tools = [tool]

# 将 tools 绑定到 LLM 上
llm = init_chat_model("openai:gpt-4.1")
llm_with_tools = llm.bind_tools(tools)


# 定义主要节点
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

# 新增 ToolNode 节点
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

# 配置条件边：判断是否调用工具
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# 任意时工具调用完都返回 chatbot 继续
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()


# 运行（简易 REPL）
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


if __name__ == "__main__":
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input)
        except:
            # 没有交互输入环境时的兜底
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input)
            break
