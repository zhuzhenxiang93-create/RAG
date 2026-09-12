# LegalMind processed_v2 数据卡

- 版本：2.0（本项目内部版本）
- 任务：匿名刑事案件事实的多标签罪名分类
- 语言：中文
- 状态：历史派生数据，未声明人工复核

## 来源

输入文件为服务器已有 `data/train_data.jsonl`，154,592 条，SHA256 `5d287aed962c8387f751c0fdd0599ea105469dfb0a0f80a89ce239373853e404`。内容来自 CAIL2018 历史派生流程，但当前服务器未保留可核验下载记录和许可证文件，故 provenance 为 `legacy_local_file_unverified`。

派生文件包含事实、罪名和量刑信息，不包含原始 `relevant_articles`。本数据卡不声称它可对外再分发。正式发布必须用官方原始数据重建。

## 构建结果

| split | 数量 |
|---|---:|
| train | 120,393 |
| validation | 15,032 |
| test | 15,097 |

标签数 202。标签映射只由 train 建立。精确文本、NFKC/空白规范化文本和 dedup group 的跨 split 重叠均为 0。

## 处理

Unicode NFKC、控制字符与空白清理、低信息/过短样本过滤、目标结论遮蔽、精确与规范化重复合并、SimHash 近重复报告、案件族分组切分、固定 seed 和文件 SHA256。

## 限制

- 原始下载来源与许可证待恢复；
- 法条字段缺失；
- 自动泄漏遮蔽命中比例高，需要人工抽检；
- 公开数据可能包含标签噪声；
- v2 test 未经法律专业人员人工复核，不能称为人工金标；
- 当前已归档的 processed_v2 文件生成于罚金字段修复之前，不包含 `fine`；代码已修复，后续必须以新版本目录重建，不能把旧文件误称为已保留罚金；
- 输出仅用于算法实验，不构成法律意见。

完整统计与哈希见 `data/processed_v2/` 和 `reports/data/`。

## 罚金字段修复后的重建计划

`processed_v2` 是修复前的历史产物，不能原地补写罚金。修复后的代码将罚金保存为
`labels.fine`（非负整数或 `null`），并在每个 split 的 Manifest 统计罚金覆盖率、缺失数、
零值数及金额分位数。安全重建使用 `configs/data/processed_v2_1.yaml`，输出到新的
`data/processed_v2_1/` 和 `reports/data_v2_1/`，验证通过前不替换旧数据和训练配置。
