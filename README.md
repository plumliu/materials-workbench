# Materials Workbench

材料手册工作台。共享 intake 后，图像 LangGraph 与表格 MinerU / Luna 独立运行；网页负责 Figure 人工采点、表格核验和最终组装。

**日常使用请读 [使用说明](USER_GUIDE.md)。** 启动本地网页可直接让 Codex 帮忙，默认地址为 <http://127.0.0.1:8766/>。

## 开发入口

使用 Python 3.12 与 uv：

```text
uv sync
uv run materials-workbench serve
```

唯一命令行入口是 `materials-workbench`。模型处理由 Codex 跟进：

```text
uv run materials-workbench intake "手册名"
uv run materials-workbench figures "手册名" --workers 2
uv run materials-workbench tables "手册名"
uv run materials-workbench apply-luna "手册名"
uv run materials-workbench assemble "手册名"
```

`figures` 与 `tables` 可以同时运行。Luna 由 Codex 按根目录合同派发 `gpt-6-luna`、`xhigh` 子代理；完成一个批次就可以执行 `apply-luna`，发布已就绪的逻辑表。Codex 的单图重试用 `figures "手册名" --figure Figure_编号 --retry`，默认保留已有人工 TAR。网页则可直接打开单页 PDF 手标；“智能识别当前图”经清空确认后替换该图全部标注，支持不同 Figure 并行识别。

可再生的中间 JSON、渲染图、模型请求与响应、模型导出 TAR 和组装产物允许重跑覆盖；JSON 使用原子替换，失败摘要随本次运行更新。同一来源 PDF 的 intake 可以重跑。人工修订版本、来源变化与并发操作仍需通过原有检查。

配置统一在根目录 `.env`；新建环境时复制 `.env.example` 并在编辑器填写。Git 追踪代码、文档、依赖清单和不含密钥的 `.env.example`。`pdfs/`、`figure_assets/`、`runs/`、`library/` 各自仅追踪 `.gitkeep` 占位文件，目录内的手册数据和产物均忽略；`.env`、依赖环境与缓存也不追踪。

这是 hardcut：新工作台只接受自己的目录及协议，旧项目的 CLI、manifest、状态目录不作为输入。两个算法包保留模块名，供新工作台内部调用。

## 代码与约定

- `src/materials_workbench/`：统一入口、状态、并行互斥、网页 API。
- `src/chart_annotator/`：intake、LangGraph、轴校准与空 Dataset TAR 导出。
- `src/pdf_tree_workflow/`：表格 OCR、Luna 补丁、人工修订及目录树组装。
- `front_end/`：手册总览、图像标注、表格核验。
- `vendor/wpd/`：固定版本的 WebPlotDigitizer 5.3、依赖和许可证。
- [架构与产物](docs/INTEGRATION_PLAN.md)、[Luna 合同](LUNA_TABLE_REVIEW_CONTRACT.md)、[组装合同](PDF_TREE_CONTRACT.md)。

运行检查：`uv run pytest tests -q`、`uv run ruff check src tests`。离线测试不会调用付费服务；完整手册回归需要本机 `pdfs/` 中的两本材料手册，否则自动跳过。

模型接口临时故障使用 SDK 指数退避：首次失败后最多重试 3 次（总计最多 4 次请求），默认等待约 0.5、1、2 秒并带随机抖动，优先遵循 SDK 支持的服务端重试提示。重试耗尽后保留最终完整错误（密钥脱敏）；模型输出校验继续使用既有修复流程。
