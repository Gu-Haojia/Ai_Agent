"""
Agent with SQL Checkpoint History & Time Travel

功能：
- 支持多轮对话
- 检查点历史记录（SQL持久化）
- 支持回溯到任意历史检查点继续对话
- 提供清晰接口：初始化、发送消息、获取历史、回溯

依赖：
- langchain
- langgraph
- langgraph.checkpoint.postgres
- langchain_tavily
"""

import os
from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# 说明：密钥请通过环境变量注入（OPENAI_API_KEY、TAVILY_API_KEY），不要在代码中硬编码

class State(TypedDict):
    messages: Annotated[list, add_messages]

class AgentWithTimetravel:
    def __init__(self, thread_id: str = "demo-agent", conn_str: Optional[str] = None):
        self.thread_id = thread_id
        self.conn_str = conn_str or os.getenv(
            "LANGGRAPH_PG", "postgresql://languser:langpass@localhost:5432/langgraph"
        )
        self.llm = init_chat_model("openai:gpt-4.1")
        self.tools = [TavilySearch(max_results=2)]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self._build_graph()
        self.config = {"configurable": {"thread_id": self.thread_id}}

    def _build_graph(self):
        from langgraph.checkpoint.memory import InMemorySaver
        builder = StateGraph(State)
        builder.add_node("chatbot", self._chatbot)
        builder.add_node("tools", ToolNode(tools=self.tools))
        builder.add_edge(START, "chatbot")
        builder.add_conditional_edges("chatbot", tools_condition)
        builder.add_edge("tools", "chatbot")
        # 优先尝试PostgresSaver，失败则降级为InMemorySaver
        try:
            self.memory = PostgresSaver.from_conn_string(self.conn_str).__enter__()
            self.memory.setup()
            print("[AgentWithTimetravel] 使用PostgreSQL持久化检查点。")
        except Exception as e:
            print(f"[AgentWithTimetravel] PostgreSQL连接失败，降级为内存模式: {e}")
            self.memory = InMemorySaver()
        self.graph = builder.compile(checkpointer=self.memory)

    def _chatbot(self, state: State):
        ai_msg = self.llm_with_tools.invoke(state["messages"])
        return {"messages": [ai_msg]}

    def send_message(self, user_content: str) -> str:
        """发送用户消息，返回AI回复"""
        events = self.graph.stream(
            {"messages": [{"role": "user", "content": user_content}]},
            self.config,
            stream_mode="values",
        )
        last_reply = ""
        for ev in events:
            if "messages" in ev:
                last_reply = ev["messages"][-1].content
        return last_reply

    def get_checkpoint_history(self) -> List[Dict[str, Any]]:
        """获取检查点历史"""
        history = []
        for state in self.graph.get_state_history(self.config):
            history.append({
                "num_messages": len(state.values["messages"]),
                "next": state.next,
                "config": state.config,
                "messages": state.values["messages"],
            })
        return history

    def continue_from_checkpoint(self, checkpoint_index: int) -> str:
        """从历史检查点继续对话，返回AI回复"""
        history = list(self.graph.get_state_history(self.config))
        if checkpoint_index < 0 or checkpoint_index >= len(history):
            raise IndexError("Invalid checkpoint index")
        checkpoint = history[checkpoint_index]
        events = self.graph.stream(
            None, checkpoint.config, stream_mode="values"
        )
        last_reply = ""
        for ev in events:
            if "messages" in ev:
                last_reply = ev["messages"][-1].content
        return last_reply

    def close(self):
        """关闭数据库连接"""
        if hasattr(self, "memory") and self.memory:
            self.memory.__exit__(None, None, None)

# 示例用法
if __name__ == "__main__":
    agent = AgentWithTimetravel(thread_id="demo-test")
    print("用户: 你好，帮我查查LangGraph是什么？")
    reply = agent.send_message("你好，帮我查查LangGraph是什么？")
    print("AI:", reply)
    print("\n>>> 检查点历史：")
    history = agent.get_checkpoint_history()
    for idx, item in enumerate(history):
        print(f"Checkpoint {idx}: 消息数={item['num_messages']}, next={item['next']}")
    if len(history) > 0:
        print("\n>>> 从第0个检查点回溯：")
        reply2 = agent.continue_from_checkpoint(0)
        print("AI:", reply2)
    agent.close()
