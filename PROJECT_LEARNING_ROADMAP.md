# LegalMind-RAG 项目学习路线

目标：完整理解项目设计、代码、实验结果与真实边界，能够应对大模型、NLP、RAG 和算法工程日常实习面试。

建议周期：10–14 天，每天 2–3 小时。

## 掌握程度定义

学习每个模块时，按以下三层检验自己：

1. **解释层**：能用自己的话说明为什么这样设计。
2. **追踪层**：能沿代码定位输入、处理逻辑和输出。
3. **修改层**：能现场修改、运行测试并排查错误。

数据、QLoRA、检索和默认 Pipeline 必须达到修改层；SFT、Reranker 微调和历史实验达到追踪层即可。

---

## 阶段一：建立项目全局认知

建议用时：1 天。

### 学习目标

- [ ] 能说明项目解决什么问题。
- [ ] 能说明为什么不能只做罪名分类。
- [ ] 能区分设计架构和当前默认 Demo。
- [ ] 能区分 Full、Pilot、Smoke 和未完成功能。

### 一句话定义

> 输入匿名刑事案件事实，先预测一个或多个候选罪名，再检索训练集类案和刑法条文，最后输出带证据引用、置信度和人工复核标记的结构化结果。

### 设计上的完整架构

```text
案件事实
  → 数据清洗/长文本处理
  → QLoRA 多标签分类
  → BM25 + Dense
  → RRF
  → Reranker
  → 结构化生成
  → 引用校验
  → 人工复核
```

### 当前默认 Demo 的真实架构

```text
案件事实
  → 历史 10K QLoRA 分类器
  → 案例 BM25
  → 法条 BM25
  → 确定性结构化生成
  → 引用校验
```

### 重点材料

- `README.md`
- `docs/interview/project_story.md`
- `PROJECT_INTERVIEW_AUDIT.md`

### 必须记住的数据

- [ ] train / validation / test：120,393 / 15,032 / 15,097。
- [ ] 标签数量：202。
- [ ] 历史测试子集 Micro-F1：0.8296。
- [ ] 类案库：120,393 案、128,107 chunks。
- [ ] 法条库：452 条。
- [ ] 自动测试：53 passed。

### 过关标准

- [ ] 能分别做 30 秒、1 分钟和 5 分钟项目介绍。
- [ ] 能准确说明哪些模块只做过实验、哪些已接入默认入口。
- [ ] 不看资料说出核心数据规模和指标。

---

## 阶段二：掌握代码结构和执行链路

建议用时：1 天。

### 重点文件

- `src/legalmind/pipeline/analyze_case.py`
- `src/legalmind/pipeline/core.py`
- `configs/pipeline/default.yaml`
- `src/legalmind/data/build_dataset.py`
- `scripts/train_qlora.py`
- `scripts/run_dense_reranker_experiment.py`

### 默认执行链路

```text
读取 YAML
→ build_pipeline()
→ 加载分类 Adapter
→ 加载案例 BM25
→ 加载法条 BM25
→ pipeline.analyze()
→ classifier.predict()
→ 检索案例和法条
→ deterministic_grounded_analysis()
→ validate_citations()
→ 返回 JSON
```

### 实践任务

- [ ] 找出分类器不存在时的返回状态。
- [ ] 找出案例索引不存在时的返回状态。
- [ ] 找出法条索引不存在时的返回状态。
- [ ] 找出分类没有标签超过阈值时的处理逻辑。
- [ ] 找出为什么 fallback 最多只给三个候选进入分析。
- [ ] 找出免责声明在哪里生成。

### 过关标准

- [ ] 面试官指出任意输出字段时，能定位生成它的函数。
- [ ] 能解释配置文件如何决定模型、索引和最大长度。
- [ ] 能解释默认 Pipeline 为什么没有使用 Dense、Reranker 和 SFT。

---

## 阶段三：彻底掌握数据流水线

建议用时：2 天。

### 重点文件

- `src/legalmind/data/build_dataset.py`
- `src/legalmind/data/normalize.py`
- `src/legalmind/data/leakage.py`
- `src/legalmind/data/deduplicate.py`
- `src/legalmind/data/split.py`
- `src/legalmind/data/contracts.py`
- `data/processed_v2/dataset_manifest.json`

### 完整处理顺序

```text
读取原始 JSONL
→ 统一字段格式
→ NFKC 和空白清洗
→ 丢弃空标签、过短和低信息文本
→ 检测精确重复
→ 遮蔽显式罪名结论
→ 对清洗后文本计算哈希
→ 合并规范化重复案件及标签
→ 建立多标签向量
→ 多标签分层切分
→ 检查案件族隔离
→ 写 JSONL、统计和 SHA256
```

### 必须理解的问题

#### 为什么存在标签泄漏？

案件文本可能直接出现“其行为已构成盗窃罪”或“应当以诈骗罪追究刑事责任”。不处理时，模型可能只是复制答案。

#### 为什么先去重再切分？

相同案件如果同时进入 train 和 test，模型可能通过记忆得到虚高指标。

#### 为什么使用案件族切分？

完全相同或规范化后相同的案件必须进入同一集合，不能只进行逐行随机切分。

#### 三种重复分别是什么？

- 精确重复：原始字符串完全相同。
- 规范化重复：清理 Unicode、空白等差异后相同。
- 近重复：文本不同但高度相似，使用 SimHash 生成审核候选。

#### 为什么近重复不自动删除？

SimHash 是近似方法，可能把真实的相似案件误判为重复，因此项目仅将其作为人工审核候选。

### 必须知道的代码缺口

当前标签映射是在切分前根据全部去重数据建立，并非文档所说的“只由训练集建立”。

准确表述：

> 当前代码在切分前冻结统一的 202 类标签空间。它没有将测试文本用于训练，但提前使用了全数据标签集合。严格做法应先切分，再由训练集建立映射，并显式处理验证和测试中的未知标签。

### 实践任务

- [ ] 手写简化版数据处理伪代码。
- [ ] 构造一个显式罪名泄漏文本并检查清洗结果。
- [ ] 构造两个空白不同但内容相同的案件并检查哈希。
- [ ] 找出代码如何验证案件族跨集合隔离。
- [ ] 阅读 Manifest 中全部过滤统计。

### 过关标准

- [ ] 能解释如何证明三个集合没有文本重叠。
- [ ] 能解释为什么普通随机切分不够。
- [ ] 能区分精确、规范化和近重复。
- [ ] 能指出数据流程至少一个优点和一个不足。

---

## 阶段四：彻底掌握多标签分类与 QLoRA

建议用时：2–3 天。

### 重点文件

- `src/legalmind/models/qlora.py`
- `scripts/train_qlora.py`
- `src/legalmind/models/inference.py`
- `src/legalmind/models/metrics.py`
- `src/legalmind/data/dataset.py`
- `src/legalmind/training/classifier_collator.py`
- `src/legalmind/training/long_text.py`
- `configs/model_qwen3_4b.yaml`
- `configs/classification/qwen3_4b_qlora_full.yaml`

### 多标签分类

一条案件可能同时属于多个罪名，因此标签是 multi-hot：

```text
[0, 1, 0, 1, ...]
```

每个标签独立使用 sigmoid 和 BCEWithLogitsLoss，而不是使用强制类别互斥的 Softmax。

需要掌握：

- [ ] Softmax 为什么不适合多标签任务？
- [ ] BCEWithLogitsLoss 做了什么？
- [ ] 为什么不能只报告 Accuracy？
- [ ] Micro-F1 和 Macro-F1 分别反映什么？
- [ ] 为什么长尾任务必须关注 Macro-F1？

### QLoRA

项目配置：

- 基座：Qwen3-4B。
- 量化：4-bit NF4、double quant。
- 计算类型：BF16。
- LoRA rank：16。
- LoRA alpha：32。
- LoRA dropout：0.05。
- target modules：`all-linear`。
- 分类头：`score` 模块单独保存。

核心解释：

> QLoRA 将冻结的基座权重以 4-bit NF4 存储，前向计算使用 BF16，只训练插入线性层的 LoRA 参数和分类头，从而降低权重、梯度和优化器状态的显存占用。

必须理解：

- [ ] 基座模型为什么被冻结？
- [ ] 4-bit 量化主要减少什么内存？
- [ ] LoRA 低秩矩阵解决什么问题？
- [ ] rank、alpha 和 dropout 分别控制什么？
- [ ] QLoRA、普通 LoRA 和全参数微调有什么区别？

### Batch 与显存

```text
有效 batch = 单卡 micro batch × 梯度累积步数 × GPU 数量
```

实验结果：

| micro batch | accumulation | 有效 batch | 峰值显存 | 结果 |
|---:|---:|---:|---:|---|
| 2 | 16 | 32 | 9.30 GiB | 完成 |
| 4 | 8 | 32 | 15.24 GiB | 完成，最终选择 |
| 8 | 4 | 32 | 21.72 GiB 后继续申请失败 | OOM |

### 动态 Padding 与长度分桶

- 固定 2,048 Padding Ratio：83.46%。
- 随机动态 Padding：46.71%。
- 动态 Padding + 长度分桶：1.39%。

动态 Padding 只补到批内最长序列；长度分桶让相近长度样本进入同一批次。这不是样本拼接，不同样本不会看到彼此 token。

### 长文本策略

- Head：只保留开头。
- Head+Tail：开头 2/3、结尾 1/3。
- Sentence：按关键词和句子位置选择。

三种策略已实现，但尚未完成 Full 消融，因此不能声称 Head+Tail 已被证明最优。

### 标签阈值

验证数据支持足够时，每个标签在 0.10–0.90 范围内搜索最佳 F1 阈值；支持不足时保持 0.5。

无标签过阈值时：

```text
选择最高概率候选
→ used_fallback = true
→ uncertain_below_threshold
→ requires_manual_review = true
```

### 指标边界

- 0.8296：历史 10K/2K 覆盖优先独立测试子集。
- 0.8658：v2 Full checkpoint-1000 的 validation 中间结果。
- 0.8658 不是最终完整测试成绩。

### 实践任务

- [ ] 手算一个两标签 BCE/阈值预测例子。
- [ ] 手算一个小样本 Micro-F1。
- [ ] 从代码中找到 QLoRA 的全部参数。
- [ ] 从代码中找到动态 Padding 的统计方法。
- [ ] 从代码中找到 checkpoint 恢复逻辑。
- [ ] 解释为什么 batch 8 OOM 后选择 batch 4。

### 过关标准

- [ ] 能清楚解释多标签 BCE 与 Softmax 的区别。
- [ ] 能解释 QLoRA 为什么节省显存。
- [ ] 能解释 Padding Ratio 优化来源。
- [ ] 能准确区分 0.8296 和 0.8658。

---

## 阶段五：彻底掌握检索系统

建议用时：2 天。

### 重点文件

- `src/legalmind/retrieval/lexical.py`
- `src/legalmind/retrieval/embedding.py`
- `src/legalmind/retrieval/index.py`
- `src/legalmind/retrieval/fusion.py`
- `src/legalmind/retrieval/retriever.py`
- `src/legalmind/retrieval/reranker.py`
- `src/legalmind/retrieval/chunker.py`
- `src/legalmind/retrieval/knowledge.py`

### BM25

需要理解：

- TF：词项在文档中的出现次数。
- IDF：越稀有的词项区分度越高。
- 文档长度归一化：避免长文档天然得分更高。
- 项目使用中文字符 bigram，避免依赖额外中文分词器。
- 项目只对包含 query term 的文档计算分数。
- 最多使用 64 个高信息 query terms。

### Dense 检索

```text
query → embedding
document → embedding
向量归一化
内积 ≈ cosine similarity
FAISS IndexFlatIP 搜索
```

需要掌握：

- [ ] Dense 为什么能召回词面不同但语义相近的案件？
- [ ] 为什么金额、工具和专有词可能更适合 BM25？
- [ ] 为什么归一化后内积可以表示余弦相似度？
- [ ] 为什么 `IndexFlatIP` 属于精确检索？

### RRF

BM25 分数与向量相似度的数值范围不同，因此不直接相加，而是融合排名：

```text
score(d) = Σ 1 / (k + rank)
```

项目默认 `k=60`，并可根据分类候选罪名做软性 boost。

### Reranker

```text
全库
→ BM25/Dense 召回 Top-N
→ RRF
→ CrossEncoder 对 Top-20 联合打分
→ 按 case_id 去重
→ 返回 Top-5
```

召回模型负责快速获得候选；Reranker 对少量 query-document pair 做精细判断。

### 检索指标

- Recall@K：全部相关文档中召回了多少。
- HitRate@K：Top-K 是否至少命中一个相关文档。
- MRR@K：第一个相关结果出现得有多靠前。
- nDCG@K：考虑排名位置和分级相关性的整体质量。

共享罪名代理 qrels 会产生非常大的相关集合，因此 Recall@10 很低；只要 Top-10 命中一条同罪名案件，HitRate 就会很高。

### Pilot 结果

| 方法 | Recall@10 | MRR@10 | nDCG@10 | HitRate@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.0061 | 0.8520 | 0.7848 | 1.0000 |
| Dense | 0.0076 | 1.0000 | 0.9863 | 1.0000 |
| Hybrid | 0.0069 | 0.9700 | 0.9013 | 1.0000 |
| Hybrid + Reranker | 0.0072 | 0.9800 | 0.9467 | 1.0000 |

正确结论：

> Reranker 改善了 Hybrid，但仍没有超过 Dense。当前 qrels 是共享罪名代理标签，尚不能推广为真实法律相关性结论。

### 实践任务

- [ ] 手算两个简短排名列表的 RRF。
- [ ] 找到中文 bigram 的实现。
- [ ] 找到 FAISS 索引类型。
- [ ] 找到按 `case_id` 去重的实现。
- [ ] 解释为什么 validation/test 案件不能进入知识库。
- [ ] 阅读检索 Pilot 指标文件。

### 过关标准

- [ ] 能解释召回与重排的职责区别。
- [ ] 能解释 Hybrid 为什么不一定超过 Dense。
- [ ] 能解释 Recall 低而 HitRate 高的原因。
- [ ] 能说明代理 qrels 与人工金标的区别。

---

## 阶段六：掌握生成、引用约束和降级

建议用时：1 天。

### 重点文件

- `src/legalmind/generation/schemas.py`
- `src/legalmind/generation/structured.py`
- `src/legalmind/generation/prompts.py`
- `src/legalmind/generation/citation_validator.py`
- `src/legalmind/generation/simple_sft.py`
- `src/legalmind/training/train_generator.py`

### Schema 约束

Pydantic 检查：

- 字段是否存在。
- 字段类型是否正确。
- confidence 是否为 high/medium/low。
- similar cases 是否包含 case ID 和理由。

### 引用约束

模型输出的案例 ID 和法条号必须存在于当次检索上下文。

它可以防止：

- 编造不存在的案例 ID。
- 引用当前没有检索到的法条。

它不能保证：

- 检索案例是否真的具有法律相关性。
- 法条版本是否仍然有效。
- 最终法律判断是否正确。

### 降级状态

- 无分类器：`degraded_no_classifier`。
- 无案例索引：`degraded_no_index`。
- 无法条索引：`degraded_no_statute_index`。
- 分类低于阈值：`uncertain_below_threshold`。
- 缺少证据：low confidence + manual review。

### 默认生成与 SFT

默认 Pipeline 使用确定性结构化模板，不是 SFT 模型。

SFT 部分需要掌握：

- 数据如何构造。
- assistant-only loss 的目的。
- 为什么要求严格 JSON。
- v3.1 为什么增加信息不足样本。
- 为什么 repair 后指标不能作为主要指标。
- 为什么当前不适合直接做 DPO/GRPO。
- v3.1 Pilot 尚未完成。

### 实践任务

- [ ] 构造一个非法案例 ID 并验证引用检查失败。
- [ ] 构造一个非法法条号并验证引用检查失败。
- [ ] 找出默认模板如何决定 confidence。
- [ ] 找出如何抽取 LLM 输出中的 JSON 对象。
- [ ] 找出中文字段如何被规范化为英文 Schema 字段。

### 过关标准

- [ ] 能解释 Schema 正确与事实正确的区别。
- [ ] 能举出引用校验仍然无法解决的问题。
- [ ] 能解释为什么默认使用确定性生成基线。
- [ ] 能准确说明 SFT 的完成状态。

---

## 阶段七：掌握实验管理、测试和项目局限

建议用时：1 天。

### 重点文件

- `src/legalmind/experiments/manifest.py`
- `artifacts/experiments/experiment_index.csv`
- `tests/`
- `reports/build_status_20260818.md`
- `reports/classification/full_training_recovery.md`

### 实验记录内容

- 模型与配置。
- 数据文件 SHA256。
- 随机种子。
- 训练和验证行数。
- 软件与 GPU 环境。
- 耗时和峰值显存。
- 指标。
- Checkpoint。
- completed/running/interrupted/failed 状态。

### 全量分类训练真实状态

```text
计划总步数：7526
已有 checkpoint：500、1000
step 1000 validation Micro-F1：0.8658
step 1001：KeyboardInterrupt
当前训练进程：无
最终完整 test：未运行
```

Manifest 仍显示 `running` 是工程缺陷。理想实现：

```python
try:
    train()
    status = "completed"
except KeyboardInterrupt:
    status = "interrupted"
except Exception:
    status = "failed"
finally:
    write_manifest(status)
```

### 项目真实局限

- [ ] CAIL 派生数据来源和许可证证据未闭环。
- [ ] 标签映射在切分前建立。
- [ ] 检索 qrels 未经人工审核。
- [ ] 法条自动解析结果未经专业人员逐条审核。
- [ ] Full 分类训练未完成。
- [ ] Full test 未运行。
- [ ] Dense/Reranker/SFT 未接入默认 Pipeline。
- [ ] v3.1 SFT Pilot 未完成。
- [ ] 没有生产级 API、监控和部署链路。

### 实践任务

- [ ] 找到每个主要指标的来源文件。
- [ ] 检查实验目录如何避免覆盖。
- [ ] 阅读至少三个 run manifest。
- [ ] 阅读一次失败实验日志并复述原因。
- [ ] 写出状态自动更新的修改方案。

### 过关标准

- [ ] 能解释 SHA256 对可复现性的意义。
- [ ] 能解释为什么中间验证结果不能作为最终测试结果。
- [ ] 能列出至少五个真实局限。
- [ ] 能给出合理的后续工作优先级。

---

## 阶段八：面试模拟与现场修改

建议用时：2 天。

### 不看资料讲解

每天至少练习一次：

- [ ] 30 秒项目简介。
- [ ] 1 分钟项目简介。
- [ ] 5 分钟完整介绍。
- [ ] 10 分钟技术深挖。

录音后检查：

- [ ] 有没有混淆 validation 和 test？
- [ ] 有没有把 Pilot 说成 Full？
- [ ] 有没有把代理 qrels 说成人工金标？
- [ ] 有没有把默认确定性模板说成 SFT？
- [ ] 有没有只报指标却无法解释原因？

### 现场代码追踪

随机抽取并在五分钟内定位：

- [ ] fallback 在哪里实现？
- [ ] RRF 的 `k=60` 在哪里配置？
- [ ] 为什么输出最多五个类案？
- [ ] 如何防止实验目录覆盖？
- [ ] 标签阈值在哪里加载？
- [ ] Padding Ratio 如何统计？
- [ ] Checkpoint 如何恢复？
- [ ] 测试集是否进入 SFT？
- [ ] 测试集是否进入案例知识库？

### 小修改练习

- [ ] 将 fallback 候选从 3 改为 2，并修改测试。
- [ ] 给 Pipeline 增加 `generation_status` 字段。
- [ ] 为引用校验增加非法法条测试。
- [ ] 修改 RRF label boost 并观察测试。
- [ ] 给 Manifest 增加 `interrupted` 状态。
- [ ] 将标签映射改为只由 train 建立。
- [ ] 写一个最小 FastAPI `/analyze` 接口。

至少实际完成前三项。真正熟悉代码的标准不是“读过”，而是“改过、运行过、遇到过错误并修复过”。

---

## 12 天学习安排

| 天数 | 学习内容 | 完成 |
|---|---|---|
| 第 1 天 | 项目故事、真实边界、指标记忆 | [ ] |
| 第 2 天 | 仓库结构和默认执行链路 | [ ] |
| 第 3 天 | 数据清洗、泄漏遮蔽、字段规范化 | [ ] |
| 第 4 天 | 去重、案件族切分、Manifest | [ ] |
| 第 5 天 | 多标签任务、BCE、Micro/Macro-F1 | [ ] |
| 第 6 天 | QLoRA、量化、LoRA 参数、训练脚本 | [ ] |
| 第 7 天 | 长文本、Padding、分桶、阈值和推理 | [ ] |
| 第 8 天 | BM25、Chunk、知识库 | [ ] |
| 第 9 天 | Dense、FAISS、RRF、Reranker 和指标 | [ ] |
| 第 10 天 | 生成、Schema、引用校验、SFT | [ ] |
| 第 11 天 | Manifest、测试、失败实验和项目局限 | [ ] |
| 第 12 天 | 模拟面试、代码追踪和现场修改 | [ ] |

### 每日时间分配

```text
40 分钟：学习原理
50 分钟：沿代码追踪
40 分钟：运行或修改代码
30 分钟：不看资料口述
20 分钟：整理问题卡片
```

---

## 最终自测题

能够不看资料回答下列问题，即基本达到面试要求：

- [ ] 1. 为什么项目不是普通单标签分类？
- [ ] 2. 如何防止罪名泄漏和同案跨集合？
- [ ] 3. 为什么使用 BCE，而不是 Softmax？
- [ ] 4. QLoRA 为什么节省显存？
- [ ] 5. 动态 Padding 为什么能从 83.46% 降到 1.39%？
- [ ] 6. 为什么最终选择 micro-batch 4？
- [ ] 7. BM25、Dense 和 Reranker 分别解决什么问题？
- [ ] 8. 为什么 Dense 反而优于 Hybrid？
- [ ] 9. Recall 很低但 HitRate 很高是否矛盾？
- [ ] 10. 引用校验能否彻底解决幻觉？
- [ ] 11. 0.8296 和 0.8658 分别代表什么？
- [ ] 12. 当前项目最大的三个不足是什么？

---

## 学习重点比例

```text
数据工程         25%
QLoRA 与分类     30%
检索系统         25%
生成与引用校验   10%
工程与实验管理   10%
```

对于日常实习，不需要成为法律专家，也不需要研究所有第三方库的内部源码。最终目标是：

> 任何简历数字都能解释来源，任何核心模块都能定位代码，任何重要设计都能说出优点、缺点和替代方案，并且能现场完成小范围修改和测试。
