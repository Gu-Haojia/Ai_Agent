import time
from agent_with_timetravel import AgentWithTimetravel

def test_agent_workflow():
    agent = AgentWithTimetravel(thread_id=f"test-{int(time.time())}")
    # 1. 发送消息
    reply1 = agent.send_message("什么是LangGraph？")
    print("AI回复1:", reply1)
    assert isinstance(reply1, str) and len(reply1) > 0

    # 2. 再发一条消息
    reply2 = agent.send_message("请用一句话总结LangGraph的用途。")
    print("AI回复2:", reply2)
    assert isinstance(reply2, str) and len(reply2) > 0

    # 3. 检查点历史
    history = agent.get_checkpoint_history()
    print("检查点历史条数:", len(history))
    assert len(history) >= 2

    # 4. 回溯到第一个检查点
    reply3 = agent.continue_from_checkpoint(0)
    print("回溯后AI回复:", reply3)
    assert isinstance(reply3, str) and len(reply3) > 0

    agent.close()
    print("测试通过！")

if __name__ == "__main__":
    test_agent_workflow()
