# Luna 表格复查合同

合同版本：**2.0**。模型：**gpt-6-luna**，推理等级：**xhigh**。

## 任务与权限

你由 Codex 主代理派发，复查本批 `input_manifest.json` 分配的逻辑表，同时检查这些来源页的公式转写，尤其是幂次。每张逻辑表只有一个负责代理，跨页表的全部来源页归同一代理。可以反复查看、放大和自检，直到可靠完成。

主代理提供合同快照、批次输入清单和唯一输出目录。输入全部只读；只在本批输出目录写补丁与报告。不修改 XLSX、候选、来源 PDF，不运行 OCR、应用补丁或组装。路径缺失或输入不符时报告主代理。

## 先读什么

1. 读取 `input_manifest.json`。只接受 `contract_version: "2.0"`。
2. 各 `assigned_candidates` 含 `candidate_id`、`base_revision`、`table_directory`、`source_physical_pages`、`sources`。`sources` 给出每页的 `page`、`pdf` 和 `image` 绝对路径。
3. 先读各表 `review_request.json`，查看 `sources` 中全部原页图像。必要时才读 `candidate.json`、`notes.md`、相应 OCR 内容。不要遍历整本手册。

行列地址从 1 开始；逻辑网格单元格编号为 `R001C001`。多个 component 保留独立列结构；不要为了齐列而伪造空列。跨页续表须核对表头、行序、页码和片段归属。

## 必须核对

- 表号、标题、所有行列、合并单元格、单位、脚注及紧邻注释。
- 数值、正负号、小数点、上下标、缺失标记；空白或破折号不能改成 0。
- `issues` 中每个 `review` / `error` 项都须有判断与理由。
- 按原页视觉检查公式转写：`10^{-3}` 与 `10^3`、负号、上标分组、指数括号、上下标混淆、转义残片。来源页本身清晰不代表 OCR 或 PDF 文本提取正确。
- 表内明确错误用补丁修正；表外公式问题只记录到批次报告，由主代理处理。无相关提取文本或来源不可辨时说明不确定，不猜写。

## 每表输出一个补丁

路径：`output/patches/<candidate_id>.json`。以下字段全部必需：

```json
{
  "schema_version": 1,
  "contract_version": "2.0",
  "batch_id": "batch_001",
  "candidate_id": "从清单原样复制",
  "table_directory": "从清单原样复制绝对路径",
  "decision": "patch",
  "reviewed_source_pages": [3],
  "issue_resolutions": [{"issue_code": "numeric_ocr", "resolution": "corrected", "reason": "源页该单元格为 5"}],
  "operations": [{
    "op_id": "op_001", "op": "set_cell", "target": {"cell_id": "R003C002"},
    "before": "S", "after": "5", "confidence": "high",
    "reason": "源页该单元格为 5",
    "evidence": [{"physical_page": 3, "description": "第三行，Value 列"}]
  }],
  "unresolved": []
}
```

`decision`：`accept` 表示无修改且无未决项；`patch` 表示有修改且无未决项；`partial` 必须有未决项。`reviewed_source_pages` 必须完整覆盖本表来源页。

`issue_resolutions` 恰好覆盖所有机器 review/error issue code，不能漏项或重复。`resolution` 为 `confirmed`、`corrected`、`false_positive` 或 `unresolved`，每项有简短理由。未解决项使用 `partial`，并在 `unresolved` 提供同 code 的记录。

每个 operation 有唯一 `op_id`、准确的 `before`、`after`、页码证据、理由及 `high` 或 `medium` 置信度。不输出低置信度修订。operations 依序应用，后续地址及旧值应反映前面的操作。

| op | target | before → after |
| --- | --- | --- |
| `set_cell` | `{"cell_id":"R003C002"}` | 旧字符串 → 新字符串 |
| `insert_row` | `{"row_index":3}` | null → 新行字符串数组 |
| `delete_row` | `{"row_index":3}` | 原行数组 → null |
| `insert_column` | `{"column_index":2}` | null → 按行排列的新列数组 |
| `delete_column` | `{"column_index":2}` | 原列数组 → null |
| `merge_cells` / `unmerge_cells` | `{"range":"A1:B2"}` | 原合并状态布尔值 → 新状态 |
| `set_metadata` | `{"field":"caption"}` | 旧值 → 新值；field 限 caption、footnotes、node_id_raw |
| `add_note` | `{"section":"notes"}` | null → 备注字符串 |
| `replace_grid` | `{"grid":"entire"}` | `{"rows":行数,"columns":列数}` → `{"rows":[[字符串]],"merges":["A1:B2"]}` |

`replace_grid` 仅在逐项补丁不合理时用。多 component 表仅支持 `set_cell`、`set_metadata`、`add_note`；逻辑行必须与 components 顺序拼接一致。需要改来源页、片段归属或组件结构时，用 `partial` + `structural_reparse_required` 说明证据，交主代理重解析。

`unresolved` 每项为 `{"code":"…","description":"…","evidence_pages":[3],"consequence":"哪些内容仍不完整"}`。仅源证据不足或必要结构超出操作范围时使用；JSON 格式错误、旧值不匹配应继续自检修复。

## 最后写批次报告

所有补丁完成后写 `output/batch_report.json`，主代理据此判断批次就绪：

```json
{
  "schema_version": 1,
  "contract_version": "2.0",
  "batch_id": "batch_001",
  "results": [{"candidate_id": "从清单复制", "patch_file": "patches/对应文件.json"}],
  "formula_review": [{"physical_page": 3, "status": "clean", "findings": []}]
}
```

`results` 恰好覆盖清单候选。`formula_review` 恰好覆盖本批不同来源物理页，每页一次：

- `no_formula`：未见公式；`clean`：对照后未见错误。二者 findings 为空。
- `uncertain`：findings 为空，另加 `note` 解释证据不足。
- `issue`：findings 非空，每项包含 `location`（页内位置）、`artifact`（出错提取文件）、`observed`（原提取文字）、`source_visible`（原页可见内容）、`suggested_latex`、`kind`（如 exponent_sign）、`candidate_id`（表外为 null）。

提交前检查全部来源页、旧值、issue 覆盖、JSON 格式和候选集合。结束时只简报 accept / patch / partial 数量、公式问题页数和输出目录。

## 主代理派发

显式使用 `gpt-6-luna`、`xhigh`，提示词只填写本批合同快照、input_manifest 和 output 的绝对路径，不重复整份合同。完成一批即可执行 `materials-workbench apply-luna 手册名`；缺少报告的其他批次继续等待，已应用批次不会重复应用。
