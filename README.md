# RouteRAG：基于 LoRA 查询路由的企业知识检索实验

[![CI](https://github.com/zhuzhenxiang93-create/RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/zhuzhenxiang93-create/RAG/actions/workflows/ci.yml)

本项目研究一个明确问题：

> 企业知识分散在邮件、聊天、研发任务、会议纪要等不同来源中。相比对所有问题使用固定检索范围，LoRA 能否学习“问题类型 + 数据来源 + 检索深度”，进而改善相关文档召回并减少无效检索？

主数据集为 [EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench)。
它模拟 Redwood Inference 公司的多源内部知识库，包含约 50 万份文档和 500 道评测问题。

项目重点是 LoRA 微调、严格数据划分和对照实验。原有文档上传页面、法律分类插件和
MASSIVE 意图分类代码保留为历史工程能力，但不再作为项目主线。

## 方法

```text
问题
  → Qwen + LoRA 结构化查询路由
  → 预测问题类型、目标数据源、跨文档需求和检索深度
  → BM25 / Dense / RRF
  → 相关文档
  → Recall@K、MRR、nDCG 与延迟评测
```

LoRA 输出示例：

```json
{
  "question_type": "project_related",
  "sources": ["linear", "slack", "gmail"],
  "multi_document": true,
  "conflict_check": false,
  "completeness_required": true,
  "retrieval_depth": "deep",
  "answerability": "answerable"
}
```

LoRA 不负责记住企业事实，也不直接生成最终答案。企业知识保留在检索库中，LoRA
只学习相对稳定的查询理解和检索策略。

## 数据隔离

官方 Redwood `questions.jsonl` 只能用于最终评测，不能拆分后训练。

```text
独立生成的训练公司 questions.jsonl
  └── LoRA train / validation

官方 Redwood questions.jsonl
  └── benchmark only
```

`prepare_enterprise_router.py` 会检查：

- 训练文件和评测文件是否为同一路径；
- `question_id` 是否重复；
- 规范化问题文本的 SHA-256 是否重复。

样例文件 `data/samples/enterprise_router_train.sample.jsonl` 仅用于测试训练链路，
不能用于报告模型效果。

## 1. 安装

Lite 工程测试：

```powershell
python -m pip install -r requirements-lite.txt
```

LoRA 和真实向量模型：

```powershell
python -m pip install -r requirements-full.txt
```

QLoRA 的 4-bit 模式还需要受支持的 CUDA 与 bitsandbytes 环境。

## 2. 下载官方评测问题

```powershell
python scripts\download_enterprise_rag_bench.py `
  --output data\enterprise_rag_bench
```

文档体积较大，建议从官方 Release 或 Hugging Face 先下载某个来源的切片，再逐步扩大。
解压后统一放置为：

```text
data/enterprise_rag_bench/corpus/
├── slack/*.txt
├── gmail/*.txt
├── github/*.txt
├── jira/*.txt
└── ...
```

## 3. 准备 LoRA 数据

先使用 EnterpriseRAG-Bench 官方生成框架创建一个独立训练公司，然后执行：

```powershell
python scripts\prepare_enterprise_router.py `
  --train-questions D:\datasets\training-company\questions.jsonl `
  --benchmark-questions data\enterprise_rag_bench\benchmark\questions.jsonl `
  --output data\enterprise_router `
  --validation-ratio 0.15 `
  --seed 42
```

仅验证代码链路时，可以把样例训练集作为 `--train-questions`，但仍必须使用不同的
benchmark 文件。

## 4. 训练 LoRA

```powershell
python scripts\train_enterprise_router_lora.py `
  --data-dir data\enterprise_router `
  --base-model Qwen/Qwen2.5-1.5B-Instruct `
  --output artifacts\enterprise-router-lora `
  --rank 16 `
  --alpha 32 `
  --epochs 3 `
  --learning-rate 2e-4 `
  --batch-size 2 `
  --gradient-accumulation-steps 8 `
  --bf16 `
  --load-in-4bit
```

首次只验证显存、依赖和数据链路：

```powershell
python scripts\train_enterprise_router_lora.py `
  --data-dir data\enterprise_router `
  --output artifacts\enterprise-router-smoke `
  --smoke-max-steps 2
```

Smoke 结果不能作为正式实验指标。

## 5. 路由器对照实验

基础模型零样本：

```powershell
python scripts\evaluate_enterprise_router.py `
  --data data\enterprise_router\benchmark.jsonl `
  --base-model Qwen/Qwen2.5-1.5B-Instruct `
  --load-in-4bit `
  --offline `
  --output artifacts\base-router.json
```

LoRA：

```powershell
python scripts\evaluate_enterprise_router.py `
  --data data\enterprise_router\benchmark.jsonl `
  --base-model Qwen/Qwen2.5-1.5B-Instruct `
  --adapter artifacts\enterprise-router-lora\adapter `
  --load-in-4bit `
  --offline `
  --output artifacts\lora-router.json
```

路由指标包括：

- JSON 合法率；
- 完整路由严格准确率；
- 问题类型准确率；
- 数据来源 Micro/Macro-F1；
- 跨文档、冲突与完整性字段 F1。

## 6. 端到端检索实验

全数据源固定检索：

```powershell
python scripts\run_enterprise_retrieval_benchmark.py `
  --corpus-dir data\enterprise_rag_bench\corpus `
  --questions data\enterprise_rag_bench\benchmark\questions.jsonl `
  --routing all `
  --retrieval hybrid `
  --dense-backend sentence-transformers `
  --output artifacts\retrieval-all.json
```

LoRA 路由：

```powershell
python scripts\run_enterprise_retrieval_benchmark.py `
  --corpus-dir data\enterprise_rag_bench\corpus `
  --questions data\enterprise_rag_bench\benchmark\questions.jsonl `
  --routing lora `
  --router-predictions artifacts\lora-router.json `
  --retrieval hybrid `
  --dense-backend sentence-transformers `
  --output artifacts\retrieval-lora.json
```

Oracle 上界：

```powershell
python scripts\run_enterprise_retrieval_benchmark.py `
  --corpus-dir data\enterprise_rag_bench\corpus `
  --questions data\enterprise_rag_bench\benchmark\questions.jsonl `
  --routing oracle `
  --retrieval hybrid `
  --dense-backend sentence-transformers `
  --output artifacts\retrieval-oracle.json
```

`hashing-smoke` 仅用于 CPU 流程验证，不是真实语义向量模型，不能用于最终结论。

## 实验矩阵

| 实验 | 路由 | 检索 | 作用 |
|---|---|---|---|
| A | 全来源 | BM25 | 稀疏检索基线 |
| B | 全来源 | Dense | 向量检索基线 |
| C | 全来源 | BM25 + Dense + RRF | 混合检索基线 |
| D | 基础模型零样本 | Hybrid | 判断微调是否必要 |
| E | LoRA | Hybrid | 项目核心方法 |
| F | Oracle 标签 | Hybrid | 路由方法理论上界 |

最终报告至少包含 Recall@10、MRR、nDCG@10、平均检索延迟和失败案例。目前仓库
不提供虚构的提升数字，GPU 训练与全量基准结果均标记为待测。

## 测试

```powershell
python -m unittest discover -s tests -v
python -m compileall -q app scripts tests
```

## 关键文档

- `docs/enterprise-router-lora.md`：LoRA 原理、训练实现和实验规范；
- `docs/enterprise-router-interview.md`：三分钟面试讲解与追问；
- `docs/enterprise-router-experiments.md`：实验记录模板；
- `configs/enterprise_router_lora.json`：初始可复现实验配置。
