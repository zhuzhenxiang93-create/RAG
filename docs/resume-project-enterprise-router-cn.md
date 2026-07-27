# 中文简历项目经历（实验完成后使用）

## 当前可用版本：不填写未验证指标

**RouteRAG：基于 LoRA 查询路由的多源企业知识检索**

- 基于 EnterpriseRAG-Bench 设计多源企业知识检索实验，面向 Slack、Gmail、GitHub、
  Jira、Confluence 等九类文档，构建 BM25、Dense Retrieval 与 RRF 混合召回基线。
- 基于 Qwen 指令模型和 PEFT LoRA 实现生成式查询路由，将自然语言问题转换为问题类型、
  目标数据源、跨文档需求、冲突检查和检索深度等结构化策略；通过 Prompt Loss Mask
  仅监督目标 JSON，并提供 Schema 校验与全源检索回退。
- 建立基础模型零样本、LoRA 路由、固定全源检索和 Oracle 路由对照实验，使用
  JSON 合法率、来源多标签 F1、Recall@K、MRR、nDCG 和检索延迟评估微调对下游 RAG
  的实际影响。
- 将官方 500 道 Redwood 问题严格保留为盲测集，实现问题 ID、规范化文本哈希和数据
  角色检查，防止基准问题进入训练语料；保存随机种子、数据摘要、Adapter和逐条预测，
  保证实验可复现。

## 可直接替换进简历的量化版

**RouteRAG：基于 QLoRA 的多源企业知识检索路由实验**

- 基于 EnterpriseRAG-Bench 构建覆盖 Slack、Gmail、GitHub、Jira、Confluence 等
  9 类来源的企业检索基准；从约 50 万文档中导出 45,278 文档可复现子集，覆盖
  722/722 个金标准文档，并将官方 500 题严格隔离为测试集。
- 基于 Qwen2.5-1.5B-Instruct、PEFT 和 NF4 QLoRA 实现生成式结构化路由，将
  q/k/v/o Attention 投影作为适配层，仅训练 435.8 万参数（0.2815%）；实现 Prompt
  Loss Mask、严格 Schema 解析、逐类 F1、原始预测留档和精确/模糊污染检测。
- 设计 Base、LoRA、全来源与 Oracle 对照实验；LoRA 将严格 JSON 合法率由 0% 提升至
  99.4%，但来源 Micro-F1 为 0.3237，识别出跨企业合成数据的分布偏移。
- 在相同 BM25 与 Top-10 设置下，硬路由将平均延迟从 248.27 ms 降至 145.62 ms
  （-41.3%），但 Recall@10 从 0.5827 降至 0.3844；通过 Oracle 0.7848 上界证明
  路由潜力，并据此提出同域非金文档数据生成、软路由及低置信度全源回退方案。

若简历篇幅只允许三行，优先保留前两条和最后一条。最后一条必须同时写延迟与召回，
不能只写“-41.3% 延迟”而隐藏质量损失。
