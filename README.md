# LangGraph Agents

> 一套面向 QQ 群与命令行的 LangGraph 多工具 Agent 栈：流式输出、SQL/内存检查点、NapCat Bot、每日自动任务与票务提醒一个都不少。

## ✨ 核心亮点

- **LangGraph SQL Agent**：`sql_agent_cli_stream_plus.py` 提供流式 SSE、可中断执行、自动/强制工具调用及“基于工具结论输出”策略，配合 PostgreSQL / 内存检查点可实现会话回放与时间旅行调试。
- **NapCat QQ 机器人**：`qq_group_bot.py` 结合 OneBot v11 回调、命令白名单、线程持久化及健康检查，为多个群提供安全的 @ 交互体验。
- **自动化任务**：`daily_task.py` 内置日间/夜间播报与偶像大师抽选监控，可多时段触发并向多个群广播，支持提醒与 Ticket 数据缓存。
- **工具矩阵**：内置 Tavily 搜索、Visual Crossing 天气、Google Directions/Flights/Hotels、Web Browser、Reverse Image、定时提醒等工具节点，可按需扩展。
- **可观测与可维护**：命令行 REPL 自带 `:history` / `:replay` / `:thread`，QQ Bot 通过 `.qq_group_threads.json`、`ticket_data/` 与 `logs/` 让状态可追踪；测试用例覆盖天气、多模态链路，确保改动可验证。

## 📁 项目结构

```
LangGraph/
|-- sql_agent_cli_stream_plus.py      # 增强版流式 Agent（推荐入口）
|-- sql_agent_cli_stream.py           # 标准流式 + 可暂停
|-- sql_agent_cli.py                  # 非流式基础示例
|-- daily_task.py                     # 每日播报 / Ticket 调度器
|-- qq_group_bot.py                   # NapCat / OneBot v11 QQ 机器人
|-- run_qq_group_bot.sh               # 机器人一键启动脚本（自动激活 .venv311）
|-- image_storage.py                  # 生成图像持久化抽象
|-- prompts/                          # 系统提示词模板
|-- docs/                             # 部署与使用文档（如 `lagrange_deploy.md`）
|-- src/
|   |-- agent_with_timetravel.py      # LangGraph 时间旅行调试 Agent
|   |-- chatbot.py / addtools.py      # Agent 主体与工具注册
|   |-- asobi_ticket_agent.py         # 偶像大师抽选抓取与解析
|   |-- google_* / web_browser_tool.py# 多种外部工具客户端
|   |-- visual_crossing_weather.py    # 天气工具封装
|   |-- timer_reminder.py             # 定时/提醒工具
|   `-- ...                           # 其余功能模块
|-- tests
|   |-- test_multimodal_unit.py
|   `-- test_visual_crossing_weather.py
|-- ticket_data/                      # Ticket 查询缓存（自动生成）
|-- images/ / logs/ / local_backup/   # 输出、日志与备份
|-- requirements.txt
|-- AGENTS.md                         # 协作规范
```

更多模块速览：

| 模块 | 说明 |
| --- | --- |
| `src/web_browser_tool.py` | 将 LangChain Web Browser 能力接入 Graph，提供半结构化网页解析。 |
| `src/google_reverse_image_tool.py` | 上传并比对图片，支持 NapCat 群内以图搜图。 |
| `src/visual_crossing_weather.py` | 调用 Visual Crossing API，供每日播报与 CLI 使用。 |
| `src/agent_with_timetravel.py` | 通过 checkpoint “时间旅行”快速复盘会话。 |
| `image_storage.py` | 对生成图片进行哈希、落盘、回查，支持 QQ Bot 与多模态测试。 |

## ⚙️ 快速开始

1. 安装 Python 3.11，并在仓库根目录创建专用虚拟环境：
   ```bash
   python3 -m venv .venv311
   source .venv311/bin/activate
   ```
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 复制 `.env`（或自行创建），填入 API Key / 数据库等配置。
4. （可选）启动 PostgreSQL（见下文）以启用 SQL 检查点；否则设置 `DRY_RUN=1` 走内存模式。
5. 运行你需要的入口（CLI / QQ Bot / 自动任务）。

## 🌍 环境变量速查

### Agent / LangGraph

| 变量 | 作用 | 默认 |
| --- | --- | --- |
| `OPENAI_API_KEY` | OpenAI/兼容模型密钥 | - |
| `MODEL_NAME` | 模型名，支持 `openai:gpt-4o-mini` 等 | `openai:gpt-4o-mini` |
| `SYS_MSG_FILE` | 系统提示词路径（`prompts/*.txt`） | 必填（增强版必需） |
| `LANGGRAPH_PG` | PostgreSQL 连接串，例如 `postgresql://user:pass@host:5432/db` | 空则走内存 |
| `THREAD_ID` | 默认线程 ID | `demo-plus` / `demo-sql` |
| `DRY_RUN` | 设为 `1` 时使用内存检查点 | `0` |
| `ENABLE_TOOLS` | 设为 `1` 时显式启用工具节点 | `1` |
| `TAVILY_API_KEY` | Tavily 搜索工具 Key | 可选 |

### QQ 机器人 & 自动任务

| 变量 | 作用 |
| --- | --- |
| `BOT_HOST` / `BOT_PORT` | HTTP 回调监听地址与端口（默认 `0.0.0.0:8080`） |
| `ONEBOT_API_BASE` | NapCat HTTP API 地址（默认 `http://127.0.0.1:3000`） |
| `ONEBOT_SECRET` / `ONEBOT_ACCESS_TOKEN` | 回调签名与 API Token（可选） |
| `ALLOWED_GROUPS` / `CMD_ALLOWED_USERS` | 群聊/命令白名单，逗号分隔 |
| `THREAD_STORE_FILE` | 群 → 线程 ID 映射文件，默认 `.qq_group_threads.json` |
| `DAILY_TASK` / `NIGHTLY_TASK` | 需要播报的群号（逗号分隔） |
| `DAILY_TASK_TIME` / `NIGHTLY_TASK_TIME` | HH:MM（24 小时制） |
| `TICKET_TASK` | 接收 Ticket 更新的群号 |
| `TICKET_TASK_TIME` | 单个或逗号分隔的多个 HH:MM（例：`02:05,16:05,22:05`） |
| `TICKET_TASK_PROMPT` | （可选）覆盖 Ticket 更新时给 Agent 的提示 |

> `.env` 中所有敏感信息均不会被仓库追踪，请通过环境变量或密钥管理服务注入。

## 🚀 运行方式

### CLI Agent

```bash
source .venv311/bin/activate
export SYS_MSG_FILE=$(pwd)/prompts/default.txt
python sql_agent_cli_stream_plus.py
```

- `:help` 查看命令，`:history / :replay <idx>` 用于调试，`:thread <id>` 切换线程。
- 若需要更轻量的演示，可改用 `sql_agent_cli_stream.py` 或 `sql_agent_cli.py`。

### NapCat QQ 机器人

```bash
./run_qq_group_bot.sh
# 或手动：
source .venv311/bin/activate
python qq_group_bot.py
```

- 支持 @ 机器人触发、`/switch` Prompt、`/clear` 重置线程、`/cmd` 查看指令。
- 提供健康检查：`curl http://127.0.0.1:8080/healthz`.
- `logs/`、`.qq_group_threads.json`、`ticket_data/` 会在首次运行时自动生成。

## 🗄️ PostgreSQL 持久化

LangGraph 默认读取/写入 `LANGGRAPH_PG` 指向的数据库，实现多节点之间的共享检查点、话题切换与“时间旅行”：

```bash
# Docker 示例
docker run --name langgraph-pg \
  -e POSTGRES_PASSWORD=langgraph \
  -e POSTGRES_DB=langgraph \
  -p 5432:5432 -d postgres:15

export LANGGRAPH_PG=postgresql://postgres:langgraph@127.0.0.1:5432/langgraph
python sql_agent_cli_stream_plus.py
```

macOS 亦可使用 Homebrew：

```bash
brew services start postgresql
# ... 使用完毕后
brew services stop postgresql
```

若暂未部署数据库，可设置 `DRY_RUN=1` 让检查点驻留内存（不跨进程持久化）。

## 📆 定时任务与 Ticket 监听

`daily_task.py` 暴露两个调度器：

- `DailyWeatherTask`：在 `DAILY_TASK_TIME` / `NIGHTLY_TASK_TIME` 触发，对配置的群号推送早晚播报（日期、节日、京都天气、抽选列表与机器人寄语）。
- `DailyTicketTask`：调用 `AsobiTicketQuery` 的 `check` / `update` 模式，一旦检测到新抽选立刻向群广播并可附带提醒。`TICKET_TASK_TIME` 支持多个时间点，以“02:05, 16:05, 22:05”形式配置即可。

两类任务都通过 QQ Bot 的 `_send_daily_text` 回调发送消息，可直接复用或扩展。缓存文件位于 `ticket_data/`，用于避免重复推送。

## 🔧 工具与扩展

- **搜索**：Tavily Search、Web Browser（抓取并总结页面）。
- **旅行**：Google Flights / Hotels / Directions 工具链。
- **天气**：Visual Crossing 天气查询，支持多地点、多时段。
- **票务**：Asobi Ticket 抓取 + `imas_ticket_tool` 便捷命令。
- **图像**：Reverse Image 上传 + `image_storage.py` 文件存档。
- **提醒**：`timer_reminder.py` 提供跨轮的定时提醒、清单管理。

所有工具均在 `src/addtools.py` 注册，继承自 LangChain 工具接口，便于自定义扩展。

## 🧊 酷炫玩法

- **时间旅行调试**：`AgentWithTimetravel` 可从任意 checkpoint 回放，将复杂对话拆解成 DAG 并复用旧节点。
- **群聊记忆命名空间**：每个群都绑定独立 `thread_id`，同时支持 `.qq_group_memnames.json` 持久化记忆命名空间，实现“群聊人格”。
- **多模态自检**：`test_multimodal_unit.py` 通过本地伪造的 Cross-Image 流程验证图片上传/缓存逻辑，配合 `image_storage.py` 避免重复上链。

## 🧪 测试

```bash
source .venv311/bin/activate
PYTHONPATH=$PWD pytest
```

当前测试覆盖：

- `src/test_agent_with_timetravel.py`：验证时间旅行 Agent 能够正确管理节点。
- `test_multimodal_unit.py`：校验图片工具链（下载、缓存、逆向上传）。
- `test_visual_crossing_weather.py`：确保天气工具参数与解析稳定。

## 🛠️ 运维 Tips

- `logs/`、`output.xml`、`local_backup/` 可帮助排查 NapCat 或 Agent 运行情况。
- `ticket_data/`、`.qq_group_threads.json`、`.qq_group_memnames.json` 均为运行期生成，建议在部署时加到持久卷中。
- `run_qq_group_bot.sh` 会检测虚拟环境并阻止裸跑 base 环境，确保依赖一致。
- 若需扩展工具，记得在 `addtools.py` 注册并在 README 的环境变量中补充依赖说明。

> 享受构建吧：LangGraph 让复杂流程可视化，NapCat 让群聊像控制台一样可编排——把它当成你自己的“多 Agent 控制塔”。
