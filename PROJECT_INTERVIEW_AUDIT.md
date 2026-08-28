# LegalMind-RAG 项目审阅与面试手册

审阅日期：2026-08-28（Pacific/Auckland）
远程项目：`/root/autodl-tmp/LegalMind-RAG`
审阅方式：远程只读环境核验、源码/配置/测试/报告交叉检查、现有测试重跑

## 1. 一句话结论

LegalMind-RAG 是一个面向匿名化中国刑事案件事实的研究型原型：以 Qwen3-4B QLoRA 做 202 类多标签罪名分类，以训练集案例和刑法条文做检索，再生成带引用校验、置信度和人工复核标记的结构化结果。

它最扎实的部分是数据治理、实验可追溯、显存优化和模块化评测；最需要诚实说明的部分是数据来源许可未闭环、检索金标缺失、全量训练未完成，以及默认端到端 Demo 尚未接入 Dense/Reranker/SFT。

## 2. 当前远程环境

| 项目 | 当前实测 |
|---|---|
| 系统 | Ubuntu 22.04.4 LTS，Linux 5.15，x86_64 |
| 平台 | AutoDL 容器 |
| cgroup CPU | `50000/100000`，即 0.5 核；`nproc` 显示 1 |
| cgroup 内存 | 2 GiB，无 Swap |
| 当前 GPU | 无；`torch.cuda.is_available() == False` |
| 历史实验 GPU | NVIDIA RTX 4090 D，约 24 GiB |
| 项目 Conda | `/root/autodl-tmp/conda/envs/legalmind` |
| Python | 3.11.15 |
| PyTorch | 当前 2.6.0+cu124；旧文档记录 2.5.1+cu124 |
| Transformers | 5.14.1 |
| PEFT | 0.19.1 |
| scikit-learn | 1.9.0 |
| NumPy | 2.4.6 |
| Pydantic | 2.13.4 |
| 项目总大小 | 约 9.5 GiB |
| 数据目录 | 约 2.3 GiB |
| artifacts | 约 7.2 GiB |
| 本地 Qwen3-4B 基座 | 约 7.6 GiB，位于项目目录外的 `/root/autodl-tmp/models/Qwen3-4B` |
| 当前项目进程 | 无训练、评测或推理服务运行 |

服务器常驻的只是 AutoDL 自身的 JupyterLab、TensorBoard、代理和 SSH 服务。项目没有部署为 Web API，也没有 Docker/Kubernetes/线上监控链路。

## 3. 仓库与工程结构

远程目录没有 `.git`，只有 `.source-commit=1daafc5`。因此它是一个带来源修订号的工作快照，不是可直接查看完整提交历史的 Git 仓库。

```text
LegalMind-RAG/
├── src/legalmind/          核心 Python 包
│   ├── data/               清洗、去重、泄漏遮蔽、切分、统计、校验
│   ├── models/             QLoRA 构建、推理、阈值和多标签指标
│   ├── retrieval/          Chunk、BM25、Dense/FAISS、RRF、Reranker
│   ├── generation/         Prompt、Schema、SFT 数据、JSON/引用校验
│   ├── pipeline/           默认统一推理入口
│   ├── training/           长文本、动态 Padding、SFT 训练
│   └── evaluation/         分类与生成评测
├── scripts/                数据、训练、检索、评测、审计脚本
├── configs/                数据、分类、检索、生成、Pipeline YAML
├── tests/                  53 个测试
├── data/                   原始/处理后数据、知识库、qrels、SFT、难负样本
├── artifacts/              模型 Adapter、Checkpoint、索引、指标和日志
├── reports/                数据、分类、检索、生成和环境审计
└── docs/interview/         中英文简历和面试问答
```

核心源码、脚本和测试共约 7,442 行 Python。项目采用 `src` layout、Pydantic 数据契约、YAML 配置、Manifest/哈希记录、pytest 和 Ruff。

## 4. 实际系统链路

### 4.1 研究/实验层面的完整设计

1. 输入匿名刑事案件事实。
2. NFKC/空白/控制字符清洗，并用规则遮蔽显式罪名结论。
3. 规范化文本哈希去重，以案件族为组做多标签分层切分。
4. Qwen3-4B 4-bit QLoRA 做 202 类多标签分类。
5. BM25 与 Qwen3-Embedding Dense 双路召回，用 RRF 融合。
6. Qwen3-Reranker-0.6B 对候选重排。
7. 检索刑法条文和训练集类案。
8. 生成严格 JSON，校验每个案例 ID 和法条号是否来自当前证据。
9. 低置信、无分类器、无索引或证据不足时降级并要求人工复核。

### 4.2 默认 Demo 的真实链路

默认配置当前实际是：

```text
案件事实
  -> 历史 10K QLoRA Adapter 分类
  -> 案例 BM25 + 法条 BM25
  -> 确定性模板构造结构化分析
  -> 引用集合校验
  -> JSON + 人工复核标记
```

默认入口没有加载 `HybridIndex`、`HybridRetriever`、Dense Embedding、Reranker 或生成 SFT Adapter。它们属于已经实现/运行过的独立实验模块，尚未接入默认主链路。面试中应说“完成模块化实验与可配置实现”，不要说“线上端到端链路已全部启用”。

## 5. 数据工程

### 5.1 数据来源

- 主文件：CAIL2018 历史派生文件 `train_data.jsonl`，154,592 条。
- 文件哈希：`5d287aed962c8387f751c0fdd0599ea105469dfb0a0f80a89ce239373853e404`。
- 当前状态：`legacy_local_file_unverified`；服务器上缺少可独立核验的原始下载 Manifest 和许可证闭环。
- 文件保留案件事实、罪名和量刑，但丢失原始 `relevant_articles`。
- `rest_data.jsonl` 有 748,203 条，但因来源、分布和重叠不清，仅审计、未纳入 v2。
- 旧 `test_data.jsonl` 仅 300 条且存在来源重叠，没有直接作为最终测试集。

### 5.2 清洗与切分

- 原始行：154,592。
- 精确重复：3,407。
- 规范化重复：3,408。
- 低信息：607。
- 过短：55。
- 最终总量：150,522。
- train / validation / test：120,393 / 15,032 / 15,097。
- 标签数：202；三个集合都覆盖全部标签。
- train 单标签/多标签：94,251 / 26,142。
- NFKC、空白和控制字符规范化。
- 规则遮蔽“构成某罪”“应以某罪追究”等显式答案；train/validation/test 命中 58,189/7,247/7,220。
- 以清洗后事实哈希形成 `dedup_group_id`，跨集合精确文本、规范化文本和 group overlap 均记录为 0。
- SimHash64 + 字符 trigram + 四段 LSH 只生成近重复人工审核候选，不自动删除。

### 5.3 一个需要修正的表述

文档称“标签映射只由训练集建立”，但实际 `build_dataset.py` 在切分前从全部去重记录统计标签并生成映射，然后才切分。这没有把测试文本或标签用于模型优化，但严格来说使用了全数据的标签词表信息。面试中不要坚持“映射只由 train 构建”；更准确的说法是“映射在切分前冻结，训练和评测共享同一 202 类标签空间”。后续最好改成先切分、再由 train 建映射，并显式处理验证/测试未知标签。

## 6. 多标签分类

### 6.1 模型与训练

- 基座：Qwen3-4B。
- 任务：202 类 multi-hot 多标签分类。
- 损失：`problem_type=multi_label_classification` 对应 BCEWithLogitsLoss。
- QLoRA：4-bit NF4、double quant、BF16。
- LoRA：`r=16`、`alpha=32`、dropout 0.05、`all-linear`，分类 `score` 头单独保存。
- 可训练参数：33,547,264，约占 1.4977%。
- 最大长度：2,048 token。
- 默认长文本：Head 2/3 + Tail 1/3；另实现 head-only 和关键词/句界选择。
- 动态 Padding、8 的倍数对齐、LengthGroupedSampler。
- 每标签阈值在验证数据的 0.10–0.90 网格调优；低支持标签保留 0.5。
- 无标签过阈值时输出概率最高的 fallback，同时标记 `used_fallback`。

### 6.2 显存与吞吐实验

在相同 2,048 token、有效 batch 32 的 4090D 压力测试中：

| micro batch / accumulation | 峰值 allocated | microbatch 吞吐 | 结果 |
|---|---:|---:|---|
| 2 / 16 | 9.30 GiB | 0.524 samples/s | 完成 |
| 4 / 8 | 15.24 GiB | 0.571 samples/s | 完成，最终选择 |
| 8 / 4 | 21.72 GiB | — | OOM |

2,000 条样本 Padding 审计：固定 2,048 为 83.46%，随机动态为 46.71%，动态加长度分桶为 1.39%。这是项目非常适合面试展开的工程优化点。

### 6.3 已完成指标

| 实验 | 数据范围 | Micro-F1 | Macro-F1 | 说明 |
|---|---:|---:|---:|---|
| Majority | v2 validation 15,032 | 0.0572 | — | 弱基线 |
| char TF-IDF + SGD OvR | v2 validation 15,032 | 0.8092 | — | 强传统基线 |
| 历史 QLoRA | 10K train / 2K validation | 0.8332 | 0.6072 | 覆盖优先子集 |
| 历史独立 reload test | 2K test 子集 | 0.8296 | 0.6052 | 覆盖优先抽样，不代表完整测试分布 |
| v2 Smoke | 1K/100 | 0.0073 | — | 只验证链路，不代表模型能力 |
| v2 Full checkpoint-1000 | validation 15,032 | 0.8658 | 0.6805 | 中间验证；训练随后中断 |

### 6.4 全量训练的真实状态

- 计划总步数 7,526，保存了 checkpoint-500 和 checkpoint-1000。
- checkpoint-1000 完成全 validation 评测：Micro-F1 0.8658、Macro-F1 0.6805、Precision 0.8754、Recall 0.8564、Hamming Loss 0.001629。
- 进入 step 1001 后收到 `KeyboardInterrupt`。
- 当前无训练进程，Manifest 仍错误地写着 `running`，README 的 `running_resumed` 也已过期。
- 最佳模型选择、训练完成耗时、最终阈值冻结和完整 15,097 条 test 评测均未完成。

因此，简历最稳妥的模型指标仍是 0.8296，并注明是历史 10K/2K 覆盖优先独立测试子集。0.8658 可在面试中作为“全量训练中间验证结果”补充，但不应写成最终测试成绩。

## 7. 检索与知识库

### 7.1 知识库

- 类案库只使用 v2 train：120,393 案，128,107 chunks。
- Chunk：中文句界，最多 1,024 token，128 token overlap。
- 元数据：case/chunk ID、字符偏移、罪名、法条、split、文本 SHA256。
- 刑法库：从政府公开页面自动下载并解析，452 个唯一条号。
- 法条文本未由法律专业人员逐条审核，时效性也需正式核验。

### 7.2 检索实现

- 稀疏：中文字符 bigram BM25，自建倒排表，只对命中 query term 的文档打分，并截取 IDF 较高的前 64 个词项。
- Dense：Qwen3-Embedding-0.6B，可选本地或 OpenAI-compatible API；向量 L2 归一化后用 FAISS `IndexFlatIP`。
- 融合：RRF，默认 `k=60`；可按分类候选罪名做 15% boost。
- 去重：最终按 `case_id` 去除同一案件的多个 chunk。
- Reranker：本地 CrossEncoder 或 API 两种后端。

### 7.3 Pilot 指标

5,000 chunks、100 queries、共享罪名代理相关性：

| 方法 | Recall@10 | MRR@10 | nDCG@10 | HitRate@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.0061 | 0.8520 | 0.7848 | 1.0000 |
| Dense | 0.0076 | 1.0000 | 0.9863 | 1.0000 |
| RRF Hybrid | 0.0069 | 0.9700 | 0.9013 | 1.0000 |
| Hybrid + 预训练 Reranker | 0.0072 | 0.9800 | 0.9467 | 1.0000 |

结论不是“模块越多越好”：Reranker 改善了 Hybrid，但仍未超过 Dense。这反而是一个很好的实验判断故事——依据指标保留可配置路由，而非固定堆叠。

全库 BM25 Pilot 使用 200 个 validation 查询：MRR@10 0.4654、nDCG@10 0.2846、HitRate@10 0.765。Recall@10 仅 0.0007，是因为“同罪名均相关”的代理相关集合极大，不能按普通检索指标直觉解释。

### 7.4 评测集与难负样本

- 200 个 validation 查询、1,139 个 train 候选、1,152 条 silver qrels。
- 4 个查询没有正候选。
- 人工审核 qrels：0；`reviewed_qrels.jsonl` 为空。
- 5,000 个 train 查询挖出 14,536 个难负三元组，覆盖 4,882 个查询。
- 负例类型包括同罪名不同行为、金额、结果、主观意图、共同/单独犯罪、既遂/未遂等八类。
- 所有难负样本均为 `unreviewed`。

领域 Reranker 的 1,000-triplet Pilot 最终训练完成：2,000 pair、1 epoch、390.2 秒、峰值 5.63 GiB。此前两次分别因浮点分类标签和 CUDA device assert 失败，第三版改单 logit/正确标签后成功。但尚无微调前后在同一人工 qrels 上的对照结果，不能声称微调提升了检索质量。

## 8. 生成、引用与不确定性

### 8.1 Full Schema

- `predicted_accusations`
- `relevant_articles`
- `key_facts`
- `missing_information`
- `similar_cases`
- `analysis`
- `confidence`
- `requires_manual_review`

JSON 使用 Pydantic 校验；引用检查要求 `similar_cases.case_id` 和 `relevant_articles` 都属于当次检索上下文。

### 8.2 默认生成不是 LLM

默认 Demo 的 `deterministic_grounded_analysis` 只是：取前四句作为关键事实、把前五个检索结果写成类案引用、输出固定分析说明，并依据是否缺法条/类案决定 low/medium confidence。它没有调用生成模型。因此面试时应说“默认安全基线采用确定性结构化生成”，不要说默认输出来自微调 LLM。

### 8.3 SFT 状态

- v2 Smoke：100 train / 20 validation，46.6 秒训练，峰值 9.46 GiB。
- 原始旧评测中 Prompt 与 SFT 的 JSON/Schema/Micro-F1 都为 0，主要因为模型输出中文字段、中文置信度、罪名后缀和截断。
- 后续引入 JSON 对象抽取与确定性字段规范化；20 条重评中简化 Schema 通过率为 Prompt 1.00、SFT 0.95，罪名 Micro-F1 为 0.9091/0.9048。这只能作为错误诊断，不是正式泛化结果。
- v3.1 数据：4,500 train / 550 validation，包含 500/50 条确定性“信息不足”样本；全部 `unreviewed`，test 使用为 0。
- v3.1 Pilot 计划 64 steps，只到 step 59；有 checkpoint-32，无完成 Manifest，也没有最终评测。
- DPO/GRPO 未实施，原因是 SFT、人工 qrels、法条监督和 chosen/rejected 偏好对均未达到质量门槛。

## 9. 工程质量

当前远程环境实测：

- `pytest -q`：53 passed，13.52 秒。
- `ruff check .`：通过。
- `ruff format --check .`：107 files already formatted。
- 数据、模型和实验产物记录 SHA256、配置、环境、耗时、显存和状态。
- 新实验目录拒绝覆盖；训练支持从 checkpoint 恢复。
- 分类器、案例索引、法条索引缺失时返回显式 degraded 状态。

不足：

- 无 Git 历史、CI、Dockerfile、API 服务、鉴权、监控或负载压测。
- 测试主要是单元/小样本逻辑测试，没有真实 GPU 端到端回归。
- 使用 pickle 保存/加载 BM25，不能加载不可信来源的索引。
- 默认 Pipeline 与实验级 Hybrid/Reranker/SFT 没有统一装配。
- 文档/Manifest 状态存在漂移，说明缺少自动状态闭环。
- 法律数据、法条和检索标注均未由法律专家系统审核。

## 10. 简历建议

### 推荐项目名

**LegalMind-RAG：证据约束的刑事案件多标签分类与类案检索系统**

这个名称比“智能法律分析平台”更准确，也更经得起追问。

### 推荐四条中文简历描述

1. 面向 CAIL2018 历史派生数据重构刑事案件多标签流水线，完成 NFKC 清洗、显式罪名泄漏遮蔽、规范化去重与案件族分组切分，构建 120,393/15,032/15,097 条、202 类数据集，并将跨集合规范化文本与案件族重叠降为 0。
2. 基于 Qwen3-4B 与 4-bit QLoRA 实现多标签罪名分类和断点续训，结合 2,048-token Head+Tail、动态 Padding 与长度分桶；4090D 实测选择 micro-batch 4/梯度累积 8，Padding Ratio 从固定长度的 83.46% 降至 1.39%，历史 10K/2K 覆盖实验独立测试 Micro-F1 0.8296。
3. 构建仅含训练集的 120,393 案、128,107-Chunk 类案库和 452 条刑法条文库，实现 BM25、Qwen3 Dense、RRF 与 Qwen3 Reranker；在统一 5K-Chunk Pilot 上完成检索消融，并依据 Dense nDCG@10 0.9863 高于 Hybrid/Reranker 的结果保留可配置路由。
4. 设计 Pydantic 严格 JSON、案例/法条引用集合校验、阈值回退和人工复核机制；建立实验 Manifest、数据哈希和 Bad Case 分析，完成 53 项测试与 Ruff 静态检查，并明确区分 Full、Pilot、Smoke 和未人工审核结果。

### 不要写的表述

- 不要写“全量模型测试 Micro-F1 0.8658”：这是 checkpoint-1000 的 validation 中间结果。
- 不要写“完整 RAG 已部署上线”：没有 API 服务，默认 Demo 也未接 Dense/Reranker/SFT。
- 不要写“Reranker 提升检索效果”：它只优于 Hybrid、仍低于 Dense，且 qrels 未人工审核。
- 不要写“法律专家标注/金标”：当前人工 qrels 为 0，法条自动解析也未逐条审核。
- 不要写“数据完全合规”：CAIL 派生文件来源与许可证证据未闭环。
- 不要写“标签映射只由训练集建立”：当前代码并非如此。
- 不要写“Full SFT 完成”：v3.1 Pilot 未跑完。

## 11. 90 秒面试讲法

“这个项目不是让大模型直接给法律结论，而是把刑事案件分析拆成可独立评测的分类、检索、生成和校验模块。数据侧，我处理了 15 万多条 CAIL2018 历史派生案件，通过答案泄漏遮蔽、规范化去重和案件族分组切分，得到 12 万训练、1.5 万验证和 1.5 万测试的 202 类多标签数据。模型侧，我用 Qwen3-4B 做 4-bit QLoRA，在 4090D 上通过 batch 压测、动态 Padding 和长度分桶把 Padding Ratio 从 83.46% 降到 1.39%；历史覆盖实验独立测试 Micro-F1 是 0.8296。检索侧，我只用训练集构建 12 万案件、12.8 万 Chunk 的类案库，同时做 BM25、Dense、RRF 和 Reranker 消融，结果显示 Dense 反而最好，所以没有为了架构复杂而强行堆模块。最后我用 Pydantic 和引用集合校验约束输出，任何低置信或无依据引用都触发人工复核。项目当前仍是研究原型，数据许可、人工 qrels、全量训练和默认主链路集成是明确的后续工作。”

## 12. 高频追问与答题要点

### 为什么是多标签而不是单分类？

一案可能同时涉及多个罪名，标签用 multi-hot，模型输出每类独立 sigmoid 概率，损失为 BCE；Softmax 会强制类别互斥，不适用。

### 为什么主指标用 Micro-F1？

标签极度长尾，Micro-F1 反映总体样本-标签判断能力且便于稳定比较；同时必须报告 Macro-F1 和长尾分桶，否则头部罪名会掩盖尾部问题。

### 为什么 QLoRA？

24 GiB 4090D 在 2,048 token 下难以稳定做 4B 全参数训练。QLoRA 用 NF4 冻结基座，只训练低秩 Adapter，兼顾显存、训练成本和可复现实验；项目没有证明 QLoRA 理论上一定优于 LoRA。

### 为什么 Head+Tail？

案件前部通常包含主体和主要行为，尾部常包含结果、量刑情节或结论；仅 head 截断可能丢失结果。但该选择尚未完成 Full 消融，所以应称工程假设而不是已证明最优。

### 动态 Padding 为什么提升这么大？

大部分事实远短于 2,048 token。固定 Padding 为每条样本补满；动态 Padding 只补到批内最长，再按长度分桶减少批内方差。注意这不是把多个样本拼接在一起，样本之间仍不可见。

### 为什么 Dense 比 Hybrid 好？

代理 qrels 由共享罪名定义，语义模型天然更接近这个判定；BM25 可能被程序性套话影响，RRF 也会把弱稀疏结果引入前排。结论只适用于该 Pilot，人工法律相关性 qrels 后需重评。

### Recall@10 为什么极低但 HitRate@10 很高？

代理定义把所有同罪名案件都视为相关，分母可能有成千上万条。Top-10 命中至少一条很容易，所以 HitRate 高；召回全部同罪名集合几乎不可能，因此 Recall 低。

### 如何防幻觉？

先用 Schema 限制字段，再把输出案例 ID 和法条号与当次检索证据做集合包含校验；证据不足或分类未过阈值时降置信并强制人工复核。它能阻止无来源引用，但不能证明检索证据本身法律上正确。

### 为什么不做 DPO/GRPO？

奖励和偏好数据尚不可靠。SFT 格式未稳定、人工 qrels 为 0、法条监督缺失，此时做偏好优化可能只会优化错误代理目标。先补数据和人工评测，再考虑小规模 DPO。

### 最大的失败与复盘是什么？

两个好例子：其一，全量分类恢复到 step 1000 后完成 0.8658 validation，但训练被中断且状态文件没自动更新；其二，Reranker 前两版因标签 dtype/输出形状导致 CUDA 错误，第三版修正为单 logit 后完成。回答重点应落在错误定位、状态闭环和防止把中间结果包装成最终结果。

## 13. 面试前最值得补的工作（按优先级）

1. 把 `running` 修成 `interrupted_at_step_1001`，并让训练脚本通过 `try/except/finally` 自动更新 Manifest。
2. 在有 4090D 的实例恢复 checkpoint-1000，完成训练、冻结验证策略，再一次性跑完整 test。
3. 修正标签映射构建顺序，真正只从 train 建表，并增加未知标签测试。
4. 将 Dense/RRF/Reranker 通过配置接入统一 Pipeline；明确默认路由和 fallback。
5. 完成至少 100–200 query 的双人法律相关性标注，报告一致性和人工 qrels 指标。
6. 完成 v3.1 SFT Pilot 和同一 validation 的 Prompt/SFT 对照；不要依赖 repair 后指标作为主结果。
7. 从官方 CAIL 来源重建数据并补许可证/下载 Manifest，恢复 `relevant_articles`。
8. 建立 Git 仓库、CI、Docker/环境锁定、README 状态自动生成；如果想投后端/平台岗，再补 FastAPI、健康检查和简单压测。

## 14. 最终定位

这是一个“工程和实验方法明显强于最终产品完成度”的项目。用于大模型算法、NLP、RAG、检索或模型工程岗位是合适的，卖点应放在：

- 多标签与长尾问题建模；
- 数据泄漏、重复和集合隔离；
- QLoRA 显存与 Padding 工程；
- 检索消融和反直觉结果判断；
- 引用约束、降级和人工复核；
- 用 Manifest/哈希区分真实结果与计划结果。

把它描述成“严谨的研究型端到端原型”会很有说服力；把它描述成“已上线的法律智能平台”则风险很高。
