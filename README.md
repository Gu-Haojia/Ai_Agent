# LangGraph Agents（流式 + 工具 + SQL 检查点）

本项目演示了使用 LangGraph/LangChain 构建具备网页搜索工具、检查点持久化以及流式输出能力的交互式 Agent。

- `sql_agent_cli_stream.py`: 流式 + 可暂停（Ctrl-C）+ SQL/内存检查点，自动/按需调用网页搜索工具
- `sql_agent_cli_stream_plus.py`: 增强版，多轮工具（可强制调用）、先 Tool 后 Agent 的输出顺序、强化“基于工具结果综合”
- `sql_agent_cli.py`: 非流式基础版（SQL 检查点），演示核心结构

> 说明：仓库中为本地调试方便保留了示例 API Key 与 brew 启停（请仅在本机调试使用）。在你的环境中，建议通过环境变量覆盖这两个值。

## 快速开始

### 1) 创建并激活虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) 安装依赖

```bash
pip install -r requirements.txt
```

### 3) 环境变量

必需/可选环境变量：

- `OPENAI_API_KEY`（必需，若使用 OpenAI 模型）
- `SYS_MSG_FILE`（必需，增强版 `sql_agent_cli_stream_plus.py` 需要，指向 `prompts/*.txt`）
- `TAVILY_API_KEY`（可选，启用网页搜索工具 Tavily 时建议设置）
- `LANGGRAPH_PG`（可选，使用 PostgreSQL 持久化检查点时设置）
- `MODEL_NAME`（可选，默认 `openai:gpt-4o-mini`）
- `THREAD_ID`（可选，默认 `demo-sql` / `demo-plus`）
- `DRY_RUN=1`（可选，启用内存检查点，便于无 PG 环境下快速运行）
- `ENABLE_TOOLS=1`（可选，显式启用工具）

### 4) 运行

- 增强流式版（推荐，具备多轮工具 + 强化综合）：

```bash
python sql_agent_cli_stream_plus.py
```

- 标准流式版（可暂停 Ctrl-C）：

```bash
python sql_agent_cli_stream.py
```

- 基础非流式版：

```bash
python sql_agent_cli.py
```

> macOS（可选）：脚本中包含 `brew services start/stop postgresql` 便于本地调试。若无需 PG，可设置 `DRY_RUN=1` 使用内存检查点。

## REPL 内置命令

- `:help` 显示命令
- `:history` 查看检查点历史（时间顺序），示例输出：
  - `[2] node=tools latest={...完整工具JSON...} messages=3 next=chatbot`
- `:replay <index>` 从指定检查点回放
- `:thread <id>` 切换线程 ID（不删除历史）
- `:newthread [id]` 新建线程并切换（不删除历史）；`:clear` 等效
- `:exit` 退出

## 工具调用与回答风格（增强版）

`sql_agent_cli_stream_plus.py` 的策略：

- 当用户显式请求“搜索/检索/互联网/先用工具/帮我搜/continue”等，强制调用 Tavily 搜索工具
- 工具返回后，模型不会再次调用工具，而是基于结果进行综合并给出清晰、可执行的结论
- 事件流输出顺序：
  - `Tool: Calling tool [tavily_search]`
  - `Agent: ...（流式输出总结）`
- 消息长度上限：内部 reducer 仅保留最近 20 条消息，以避免上下文无限增长。
- 对于敏感或不现实请求（例如“如何与某公众人物结婚”），模型将礼貌地解释现实限制，并提供一般性、合规的建议（而非生搬搜索结果）

## 示例对话

- 触发搜索并总结：

```
User: 搜索...
Tool: Calling tool [tavily_search]
Agent: （综合工具结果的中文总结 + 简明要点 + 参考链接）
```

- 强制先搜索再给建议：

```
User: 先用工具搜索互联网,再告诉我如何xxxxxx
Tool: Calling tool [tavily_search]
Agent: （解释现实限制 + 合理建议；不生搬工具原文）
```

- 连续搜索（多轮）：

```
User: 搜索XXX
User: 是XXXXXXXX
User: 继续
User: Search for XXXXXX
```

> 在上述多轮中，增强版会避免“跨轮误判导致不再调用工具”的问题；当用户输入“继续”且上轮 AI 承诺搜索时，会再次调用工具兑现承诺。

## PostgreSQL 持久化（可选）

将检查点存入 PG：

1. 启动本地 PG（macOS Homebrew 示例）
   ```bash
   brew services start postgresql
   ```
2. 设置连接串
   ```bash
   export LANGGRAPH_PG=***REMOVED***
   ```
3. 运行脚本（不要设置 `DRY_RUN=1`）

关闭服务：
```bash
brew services stop postgresql
```

## 目录

- `sql_agent_cli_stream.py`：流式 + 可暂停 + SQL/内存检查点
- `sql_agent_cli_stream_plus.py`：增强流式（多轮工具/强制搜索/强化综合）
- `sql_agent_cli.py`：基础（非流式）

提示：`sql_agent_cli_stream_plus.py` 需要设置 `SYS_MSG_FILE` 指向系统提示词文件，例如：
```bash
export SYS_MSG_FILE=$(pwd)/prompts/default.txt
```

---
