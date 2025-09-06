# AGENTS.md
本文件为 Codex 在此仓库中工作时的指导说明。

## 必须遵守的事项 (YOU MUST)
- 所有回答必须使用中文。
- 开始工作前必须切换到新建的分支。
- 必须按照[## 作业流程]中的步骤执行。
- 积极使用 **serena MCP**。

## 严禁的事项 (YOU NEVER)
- 不得在代码中硬编码密码或 API Key。
- 未经用户确认，不得删除任何数据。
- 不得在没有测试的情况下部署到生产环境。
- 严禁在 **main 分支**上直接作业。
- 不要执行 `git add .` 或 `git add -A`。
- 不要 add / commit / push 不必要的文件。
- 严禁使用 base 环境，必须先激活虚拟环境后再作业。
- 不要随意使用 `try-except` 隐藏错误。
- 不要写大范围的 `try-except`。
- 不要随意修改导入方式（相对改绝对或绝对改相对）。
- 避免在命令行用 parse 传入参数（debug 模式除外）。例如 `python3 文件名.py --data data` 不允许。
- 禁止直接编辑 `data/raw/` 和 `data/processed/` 目录中的文件。
- 禁止直接查看原始数据。
- 不要进行「回退(fallback)」实现，如果结果与预期不同必须抛出 `assert` 错误。
- 不要写类似 `if hasattr(x, 'values')` 的类型分支代码，会降低透明性。

## 作业流程 (MUST!!)
- 阅读 `claude.md`。
- 切换回 `main` 分支并执行 `git pull`。
- 阅读目标文件后再执行与验证结果。
- 新建分支。
- 进行修正。
- 执行 `git add`（仅 add 修改过的文件）。
- 执行 `git commit`。

## 代码要求
- 修改和新增功能必须遵循现有设计与结构。
- 积极使用 **类**，遵循面向对象思想，保证可复用性和扩展性。
- 注重设计模式。
- 明确入参和返回值类型。
- 必须撰写 **docstring**。
- 注释必须为中文。
- 错误处理必须显式进行，避免滥用 `try-except`。
- 使用 `assert` 明确前提条件。
- 所有函数和类必须显式注明参数和返回类型。

## 每次开始与结束修正时必须执行的操作
- **开始作业时**  
  - 在 `main` 分支执行 `git pull`。  
  - 从 `main` 新建专用分支（如 `feat-功能名` 或 `fix-修正名`）。  
  - 禁止在 `main` 上直接修改。
- **结束作业时**  
  - 必须提交本次修改 (`git commit`)。

## 修正时的注意点
- 修正必须确认不会影响其他流程。  
- 若其他部分也需要调整，必须保证整体功能与预期一致。

## 提交前检查清单
- 必须进行完整的运行测试。  
- 发现错误必须更新任务并修正。  
- 只有确认无误后才可提交。

## 目录与文件操作最佳实践
### 文件操作前检查
1. **读/编辑前**  
   `pwd` 确认当前目录。  

2. **运行脚本前**  
   `pwd` 确认当前目录。  

3. **执行 git 操作前**  
   `pwd && git branch && git status` 检查路径、分支、状态。

## 提交信息规范
### 格式
1. **概要行 (Subject)**  
   - 不超过 50 字  
   - 命令式，直观说明做了什么  
   - 推荐使用 Conventional Commits 格式：`type(scope): 内容`  
     - type: feat, fix, docs, style, refactor, test, chore  
     - scope: 修改位置（如 preprocess, model, api, ui 等）

2. **空行**

3. **正文**  
   - 必须用中文描述修改原因与实现方式  
   - 每行不超过 72 字符  
   - 若有关联 Issue，用 `Refs: #123`

## Pull Request 规范
PR 必须包含以下内容：
- 修改背景与目的
- 实现方式说明
- 测试方法
- 是否有破坏性修改
- 提示审核者的注意点

## Docstring 要求
- **所有函数/方法/类** 必须写 docstring。  
- 遵循 Google 风格，包含：
  1. 功能总结  
  2. Args: 参数名 + 类型 + 说明  
  3. Returns: 返回值类型 + 说明  
  4. Raises: 可能抛出的异常与条件  

示例：
```python
def compute_average(values: list[float]) -> float:
    """
    计算浮点数列表的算术平均值。

    Args:
        values (list[float]): 需要计算平均值的数值列表

    Returns:
        float: 列表中数值的平均值

    Raises:
        ValueError: 当 `values` 为空时抛出
    """
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)
```

## 绘图规范
- 图表标题、坐标轴、图例使用英文。
- 字体与数字尽量放大。
- 图例避免遮挡数据。

## Jupyter Notebook 规范
- Notebook 必须先转为 `.py` 文件再进行修改与提交。
- 转换命令：
```bash
jupyter nbconvert --to script <file.ipynb> --output <file_claude.py>
```

## 无需额外许可的操作
- 代码执行

## 需要许可的操作
- 新建分支
- Git 操作（commit, push）
- 创建 PR
- 不可逆操作
- 文件删除
- commit 撤销
- 库/包安装

---

## 项目概览
以下部分用于记录项目的专有信息。  
如执行 `/init`，请在此处填写项目概要。
