# Enterprise Router LoRA 技术说明

## 1. 为什么不是把知识微调进模型

企业知识变化快，而且答案需要引用。将文档事实微调进参数会带来知识过期、来源不可追踪
和更新成本高的问题。因此本项目将知识保留在 RAG 语料库中，只用 LoRA 学习查询路由。

研究假设是：

> 学习式路由器可以根据问题选择更合适的数据来源和检索深度，从而在保持或改善召回的
> 同时减少无效候选文档与检索开销。

这一假设必须通过对照实验验证，不能仅凭训练 Loss 下降得出。

## 2. 训练任务

模型输入一条问题，输出严格 JSON：

```json
{
  "question_type": "conflicting_info",
  "sources": ["slack", "linear"],
  "multi_document": true,
  "conflict_check": true,
  "completeness_required": true,
  "retrieval_depth": "deep",
  "answerability": "answerable"
}
```

这相当于一个生成式多任务学习问题：

- 单标签：`question_type`、`retrieval_depth`、`answerability`；
- 多标签：`sources`；
- 二分类：`multi_document`、`conflict_check`、`completeness_required`；
- 结构约束：输出必须通过 Pydantic Schema 校验。

## 3. 为什么选择生成式路由

相比单一序列分类头，生成式路由的优势是一次产生完整检索计划，后续增加字段时无需重新
设计多个分类头。代价是必须评估 JSON 合法率，并为解析失败设置回退策略。

为了避免模型学习重复输出 Prompt，训练 Collator 将 Prompt 对应的 Label 全部设为
`-100`，损失只计算目标 JSON：

```text
input_ids = prompt_tokens + target_tokens
labels    = [-100, ... -100] + target_tokens
```

## 4. LoRA 注入

初始配置对 Attention 的四个投影层注入低秩增量：

```text
q_proj, k_proj, v_proj, o_proj
```

原始权重冻结，训练：

\[
W' = W + \frac{\alpha}{r}BA
\]

默认 `r=16`、`alpha=32`、`dropout=0.05`。这些只是首轮可复现参数，不是最优结论。
正式实验应至少比较不同 Rank，记录可训练参数量、显存峰值、训练时间和验证集指标。

QLoRA 使用 NF4 量化基础模型，并保留 LoRA 参数训练。它降低显存需求，但需要受支持的
CUDA 与 bitsandbytes 环境。

## 5. 数据划分

官方 Redwood 500 道问题只用于一次最终测试。

训练数据来自 EnterpriseRAG-Bench 官方框架生成的另一家虚构企业。两者必须具有不同：

- 公司和项目名称；
- 文档与问题 ID；
- 问题文本；
- 业务事实。

仓库执行 ID 与规范化文本哈希检查。语义近似泄漏仍需人工抽查，建议额外用 Embedding
近邻检查训练集与测试集。

`data/samples/enterprise_router_train.sample.jsonl` 只有 20 条人工示例，仅验证代码。

## 6. 标签推导

输入问题集需要兼容以下字段：

```json
{
  "question_id": "trainco_0001",
  "question_type": "project_related",
  "source_types": ["linear", "slack"],
  "question": "Why was the project delayed?",
  "expected_doc_ids": ["doc_1", "doc_2"]
}
```

规则：

- `sources` 直接来自 `source_types`；
- 多个 Gold 文档或特定复杂问题类型会启用 `multi_document`；
- `conflicting_info` 启用冲突检查；
- `project_related`、`completeness` 等启用深度检索；
- `info_not_found` 标记为不可回答。

这些是可解释的实验标签，不应被包装成模型自动发现的真理。

## 7. 评测层次

### 路由层

- JSON Valid Rate
- Exact Route Accuracy
- Question Type Accuracy
- Source Micro/Macro-F1
- 各布尔字段 F1

### 检索层

- Recall@K
- MRR
- nDCG@K
- Gold Document Coverage
- 平均候选文档数
- 平均与 P95 延迟

### 关键对照

1. 全来源固定检索；
2. 基础模型零样本路由；
3. LoRA 路由；
4. Oracle Gold 路由。

Oracle 与 LoRA 的差距反映路由模型还有多少改善空间；LoRA 与全来源的比较才回答微调对
下游系统是否真正有用。

## 8. 失败与降级

硬过滤存在漏召回风险，生产策略应使用软路由：

1. 首轮检索预测来源；
2. 召回分数或证据覆盖不足时，自动扩展到全来源；
3. JSON 解析失败时使用全来源默认策略；
4. `info_not_found` 预测只能触发更严格证据阈值，不能单独决定拒答。

如果实验发现 LoRA 降低 Recall，应如实分析来源漏判、训练覆盖不足和阈值问题，不得只
报告有利指标。

