# 材料手册组装合同

版本：2.0。适用入口：`materials-workbench assemble` 与网页“组装手册”。工作目录协议见 [架构与产物](docs/INTEGRATION_PLAN.md)，Luna 只需阅读 [Luna 合同](LUNA_TABLE_REVIEW_CONTRACT.md)。

## 输入与完成门槛

- 原始 PDF：`pdfs/<手册>.pdf`，不覆盖或改写。
- 来源页：`figure_assets/<手册>/_page_review/page_####.pdf`，从第一页连续到最后一页，每文件恰好一页。
- 图像：同手册 `Figure_<编号>/Figure_<编号>.tar`，采用网页当前已保存版本。
- 表格：`runs/<手册>/tables/manifest.json`、简短 index 及固定 `Table_*` 目录中的候选、工作簿、出处、修订。
- 人工状态：Figure `review.json`；Table `human_review.json`，后者必须匹配候选 revision 与单元格旧值。

默认要求全部图像和表格确认、没有 intake 异常或 partial 表。用户明确选择允许带未完成项组装时可跳过完成门槛，但成品 `REVIEW_STATUS.json` 必须记录不完整状态。表格只应用 verified 人工修订，draft 保留在工作目录。Figure 采用已保存 TAR，其确认状态单独记录。

只接受工作台 schema 2 输入，不读取旧项目 manifest、状态目录或旧 CLI 输出。原项目资料的一次性转换不属于运行时协议。

## 成品目录

```text
library/<手册>/
  <手册>.pdf
  REFERENCES.md
  STRUCTURE_VALIDATION.json
  REVIEW_STATUS.json
  1 <一级章节标题>/
    content.md
    source_pages/physical_page_####.pdf
    1.1 <小节标题>/
      ...
    1.2 [Table] <表题>/
      content.md
      source_pages/physical_page_####.pdf
      table.xlsx
      notes.md
      table_provenance.json
      human_review.json                 # 存在已确认修订时
    1.3 [Figure] <图题>/
      content.md
      source_pages/physical_page_####.pdf
      Figure_1.3.tar
```

有引用时，节点另有 `references.md`、`links.md` 和 `related_pages/`。章节数量与层级以原 PDF 为准。

## 节点与正文

章节是目录节点，Figure 和 Table 是同级语义叶子。物理页只作来源附件，不能创建 `page_####` 业务叶子。跨页表保持一个逻辑节点，可包含多个列结构不同的 component。OCR 发现目录未列出的表时也应建立逻辑节点。

编号按原文保留，解析可识别点分数字、字母和括号后缀；`node_id_raw` 保留印刷形式，`node_id_core`、`node_id_suffixes`、`node_id_match_key` 用于匹配。不能把后缀自动改成新章节或虚构编号。无编号表 `node_id_raw` 为 null，使用稳定 `synthetic_id`，并记录 identifier_status。重复编号依来源区别，不静默合并。

Windows 文件名转义非法字符，超长标题可缩短并消除碰撞；正文元数据保留完整标题。文件名缩短不改变节点标识和关联关系。

每个节点都有 `content.md`，即使正文为空或来源缺失。最小 YAML 头为：

```yaml
node_type: section
node_id_raw: "1.1"
title_raw: "Commercial Designations"
source_physical_pages: [1]
source_printed_pages: [1]
content_status: text_extracted
data_status: not_applicable
```

正文保留原文、段落和阅读顺序；允许修复版面断行造成的断词，不翻译、不补充外部知识、不静默更正技术内容。目录条目本身可以作为来源，但不能假称叶子内容已经找到。

`content_status` 为 `text_extracted`、`empty_in_source`、`missing_source_page` 或 `extraction_failed`。`data_status` 为 `not_applicable`、`pending`、`ocr_only`、`machine_validated`、`luna_reviewed`、`tar_present`、`partial` 或 `missing`；`review_required` 表示尚须机器复查的候选状态。人工确认与机器状态分别记录。

## 页码、结构与链接

- 始终区分从 1 开始的物理页码和印刷页码。独立来源附件从 `_page_review/` 读取，不重新抽页生成第二套工作来源。
- 编号正文与原页内容是结构权威；目录文字、书签和直接链接用于定位及交叉校验。书签差异写入 `STRUCTURE_VALIDATION.json`，不能让书签覆盖正文结构。
- 支持 PDF 内部直接跳转与外部 PDF 链接；保留无法解析的链接和缺失说明。返回目录、自引用和重复链接不重复制造附件。
- Table/Figure 的直接引用写入 `links.md`，所需目标页写入 `related_pages/`。只解析本节点来源页的直接引用，不递归扩展被引用页的引用。
- 根 `REFERENCES.md` 保存完整参考文献；节点 `references.md` 保存该节点直接使用的条目。不能只保留文献编号而丢失正文。

## Table 内容

图像与表格支线共用 intake 页码映射。同页可兼有 Figure / Table；混合页与不确定页须保留为 OCR 候选，不能因 Figure 存在而丢表。

MinerU 连续页片段通过本地页索引映射到原物理页。每个原始表块保留 fragment 来源；按标题、bbox、表头及列结构判断关联，不能仅凭相邻页或相同列数合并。跨页续表归并在一张逻辑表中，同页不同列结构用 components 保留。

`table.xlsx` 每个 component 对应一个 worksheet，保持原行列、合并单元格、单位和缺失值。数值可转为数值单元格；原始文本以 `=` 起头时必须作为字符串，不能变成 Excel 公式。不能用 0 替代空白或破折号。

`notes.md` 保存表题、脚注、注释、未转写的嵌入图像和未解决事项。`table_provenance.json` 保存来源页、fragment/component、MinerU 参数、机器检查路线、Luna 补丁和人工修订摘要。

简单表通过规则检查可标 `machine_validated`；风险表由 Codex 按 Luna 合同复查后标 `luna_reviewed`；经主代理明确对照来源完成结构重解析的表可标 `source_reviewed`。不足以解决的问题保持 `partial`。结构重解析由 Codex 处理，不在人工页面强行补列掩盖。

最后人工逐项核对内容，可编辑单元格并保存稀疏修订；确认前必须核对全部来源页。组装在成品工作簿副本上应用已确认修订，不回写机器工作簿。

## Figure 内容

每个 Figure 采用当前 WPD TAR，保留 Axis、Dataset、点组、元数据和点顺序，不转成额外 Excel，不在组装时重新采点。TAR 必须可解析，并包含标准 WPD 项目数据。

一个 caption 对应一个语义 Figure。原文明示多个独立 caption 时可分别建叶子；同一个 Figure 明示多个面板时，同叶子可容纳规范命名的面板 TAR。缺少 TAR 标 missing，存在损坏或部分数据标 partial；可读 TAR 标 tar_present，不能据此推断人工已经确认。

## 发布与检查

同手册组装与后台结果发布、人工保存互斥。先生成临时成品，完成后替换 `library/<手册>`；失败保留上一版，禁止全局清理其他手册的临时目录。

检查结构、节点唯一性、来源页覆盖、原 PDF 保留、引用非递归、TAR 可解析、工作簿与组件对应、人工版本一致和完成状态。缺失或未决问题必须出现在成品状态或出处中，不能用空文件假装完成。
