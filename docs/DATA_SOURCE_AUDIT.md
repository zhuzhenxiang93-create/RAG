# Data source audit

官方仓库、论文、下载路径、许可证位置、文件哈希和程序实算统计位于
`reports/data_v3/sources`。JuDGE 仓库子集实测 2,505 条；README 的“超过 10 万”指另一份
全量语料，不能替代已审计规模。发布年份也不能替代案件判决年份。

- JuDGE：仅本地审计；MIT 文本没有明确覆盖数据再分发。
- HRN/MultiLJP：官方仓库未发现许可证，停止导入。
- MultiJustice：MIT，仅规划为多主体鲁棒性集；Git LFS payload 在服务器返回 403。
- CAIL2018：历史基准/预训练候选；无 judgment_date 字段。
- LeCaRDv2：MIT 专家 qrels 独立检索评测，不作量刑监督。
- LAIC2021/ML-LJP：官方仓库未发现许可证，停止导入。
