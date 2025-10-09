from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# 1) 定义状态（State）
class State(TypedDict):
    # 使用 add_messages 作为 reducer，确保新消息是“追加”而不是覆盖
    messages: Annotated[list, add_messages]


# 2) 创建图构建器
graph_builder = StateGraph(State)


import os
# 说明：请通过环境变量提供 OPENAI_API_KEY

# 3) 选择一个聊天模型（此处示例：OpenAI）
from langchain.chat_models import init_chat_model

# 需要你在环境里已设置 OPENAI_API_KEY
llm = init_chat_model("openai:gpt-4.1")


# 4) 定义一个节点（Node）
def chatbot(state: State):
    # 把现有消息丢给 LLM，拿到回复并追加到 messages
    return {"messages": [llm.invoke(state["messages"])]}


# 5) 把节点加入图
graph_builder.add_node("chatbot", chatbot)

# 6) 指定入口与出口
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# 7) 编译图
graph = graph_builder.compile()


# 8) 运行（简易 REPL）
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        # print('event:', event)
        for value in event.values():
            # print('value:', value)
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
