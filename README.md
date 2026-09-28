# Materials Workbench

材料手册工作台：准备 Figure 标注项目，在内嵌 WebPlotDigitizer 中人工采点，核验 OCR + Luna 表格，最后组装可追溯的手册目录树。

当前完成了仓库调研、Git 初始化、Python 3.12 环境、合并依赖安装和兼容检查。业务代码迁移及统一网页属于下一阶段，具体方案见 [整合方案](docs/INTEGRATION_PLAN.md)。

## 环境

本项目统一使用 Python 3.12 和 uv，依赖已记录在 `pyproject.toml` 与 `uv.lock`。开发者可用 `uv sync` 重建环境。

原两个项目的配置已合并到根目录 `.env`，该文件由 Git 忽略。需要新建配置时参考 `.env.example`，在编辑器中填写。现有模型接口配置已保留；原配置中的 `MINERU_TOKEN` 为空，下一次调用 MinerU 前需要填写。

原始 PDF 放在 `pdfs/`，Figure 资产放在 `figure_assets/`，处理记录放在 `runs/`，最终成品放在 `library/`。这些目录均不进入 Git。每本手册的处理记录按手册名和 Figure/Table 编号组织，不采用层层随机目录。

WebPlotDigitizer 使用 5.3 源码，固定到本次核查的官方提交；开发准备文件位于被忽略的 `.research/` 中。正式整合时将所需前端源码和许可证放入 `vendor/`，由同一个本地服务提供页面。

## 当前验证

- 两个旧项目在新环境中运行：图像项目 151 项测试通过，表格项目 49 项测试通过。
- 15 个运行依赖的主要模块均能导入；`uv pip check` 通过。
- WPD 5.3 核心代码通过 11 份项目的载入与序列化检查，包括 3 份由现有导出器生成的空 Dataset 项目和 8 份人工标注样本。
- 3 份空项目添加数据点后能够序列化、重新载入并保留数据点。统一网页的导入、保存和切换流程仍待实现及浏览器验收。
