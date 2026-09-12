# Dataset v3

`processed_v3_candidate` 是保留的 v2.1.1 数据的可审计 schema 升级。由于本地源归档、
案件日期及原始数据许可范围无法完全核验，它不会被称为正式 v3。重建命令：

```bash
python scripts/build_processed_v3.py --config configs/data/processed_v3.yaml
```

Schema 区分罚金 missing、明确 zero、positive、invalid 与 not_applicable；死刑和无期不
生成伪造月份。各任务 mask 防止缺失标签成为负样本。数据集发布日期不等于判决年份。

CAIL 衍生历史案件可作基准，但不代表当前裁判实践。模型是需要人工复核的研究原型，
不构成法律意见，不能替代法官或律师；仅凭罪名无法可靠预测个案刑期或罚金。
