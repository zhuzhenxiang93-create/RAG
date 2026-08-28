# 分类 Bad Case（已验证实例）

## 阈值以下仍返回候选

`examples/case.txt` 描述趁工作人员不备拿走手机并主动投案。历史 10K Adapter 的最高候选为“抢夺”，概率 0.8210，低于该类阈值 0.85；旧推理逻辑在没有任何标签过阈值时回退到 Top-5，却仍返回普通 `ok` 状态。

修复：`ClassificationResult` 增加 `used_fallback` 和 `max_probability`，流水线状态改为 `uncertain_below_threshold`；Fallback 只传前三候选给后续分析，并强制人工复核。

该实例也显示“盗窃/抢夺”混淆：关键差别是是否趁人不备和平静取得，适合作为后续混淆罪名定向数据，但单个示例不能证明整体改进。

## 待开展统计

- v2 Full 验证预测尚未生成，因此频次分桶、混淆矩阵和长文本截断归因均为 `not_run`。
- 历史 10K 结果只用于基线，不反向调整 v2 test。
