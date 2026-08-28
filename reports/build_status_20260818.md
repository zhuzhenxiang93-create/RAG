# LegalMind-RAG 构建与验收状态（2026-08-18）

本文只记录服务器文件、命令日志和已完成实验能够支持的结果。v2 Full QLoRA 正在运行，完成前不填写其训练时间、显存峰值或 Micro-F1。

## 1. 项目完成状态

数据 v2、分类基线、QLoRA Smoke、Batch Benchmark、法条/案例知识库、BM25、Dense/Reranker Pilot、结构化生成、SFT 数据、生成 SFT Smoke、统一推理、测试、README 和面试材料已经实现并验证。v2 Full QLoRA 状态为 `running`；全量长文本消融、全量 Reranker 训练和全量生成 SFT 尚未运行。

## 2. 环境和 GPU 状态

- 工作目录：`/root/autodl-tmp/LegalMind-RAG`
- Conda：`/root/autodl-tmp/conda/envs/legalmind`
- Python 3.11.15，PyTorch 2.5.1+cu124，Transformers 5.14.1，PEFT 0.19.1
- GPU：NVIDIA GeForce RTX 4090 D，24,564 MiB
- v2 Full 启动观察：约 10.84 GiB 显存，GPU 利用率 100%

## 3. 实际使用的数据文件

分类数据源为 `data/train_data.jsonl`。`data/local_legacy/rest_data.jsonl` 和旧 `test_data.jsonl` 仅完成审计，没有并入 v2。处理后文件位于 `data/processed_v2/`。案例知识库只使用 v2 train；validation/test 没有进入索引。

## 4. 数据来源与许可

分类文件记录为 `CAIL2018_legacy_local`，当前本机文件缺少可独立核验的原始下载 Manifest，状态为 `legacy_local_file_unverified`。公开发布前必须从 CAIL 官方来源重建并补齐许可证证据。法条来自国家统计局公开的《中华人民共和国刑法》页面，下载 URL、时间和哈希保存在 `data/raw/statutes/manifest.json`；自动解析内容仍标记为 `unreviewed`。

## 5. 原始数据数量

- `train_data.jsonl`：154,592
- `local_legacy/rest_data.jsonl`：748,203（未纳入）
- `local_legacy/test_data.jsonl`：300（未直接作为 v2 test）

## 6. 清洗后数据数量

总计 150,522。过滤统计：低信息 607、过短 55、精确重复 3,407、规范化重复 3,408。显式目标结论遮蔽数量为 train 58,189、validation 7,247、test 7,220。

## 7. 数据切分

- train：120,393
- validation：15,032
- test：15,097
- seed：42
- 方法：规范化去重后，按案件族进行迭代多标签分组切分

## 8. 标签数量和分布

共有 202 个罪名标签，三个集合均覆盖全部标签。train 中单标签 94,251、多标签 26,142。频次区间由实际训练分布确定：极低频 1–72、低频 73–1,891、高频 1,892–9,618。完整频次见 `reports/data/label_distribution.csv`。

## 9. 重复与泄漏检查

train/validation/test 的精确文本、规范化文本和 `dedup_group_id` 跨集合重叠均为 0；未知标签 0；人工审核虚假标记 0。近重复候选保留在 `reports/data/near_duplicate_candidates.json`，用于后续人工审计。

## 10. QLoRA 配置

Qwen3-4B，4-bit NF4，double quant，BF16，gradient checkpointing；LoRA `r=16`、`alpha=32`、`dropout=0.05`、`target_modules=all-linear`。训练参数 33,547,264，占总参数 1.4977%。使用 Head+Tail 2,048 tokens、动态 Padding、`pad_to_multiple_of=8` 和长度分桶。

## 11. Batch Benchmark

保持有效 batch 32：micro-batch 2/accumulation 16 峰值 9.30 GiB；4/8 峰值 15.24 GiB；8/4 峰值 21.72 GiB 后 OOM。全量配置选择 4/8。2,000 条 Padding 审计：固定长度 0.8346、随机动态 0.4671、动态+长度分桶 0.0139。

## 12. 训练时间和峰值显存

历史 10K QLoRA 为 6,898.5 秒、9.46 GiB。v2 Smoke 为 263.5 秒训练、294.9 秒端到端、9.46 GiB。v2 Full 正在运行，最终值为 `pending_training_completion`。

## 13. 分类 Micro-F1

- Majority validation：0.0572
- char TF-IDF + Linear validation：0.8092
- 历史 10K QLoRA validation：0.8332
- 历史独立 test：0.8296
- v2 QLoRA Smoke validation：0.0073，仅验证链路
- v2 Full：`pending_training_completion`

## 14. 检索指标

全训练库 BM25、200 个 validation 查询、共享罪名代理相关性：Recall@10 0.0007、MRR@10 0.4654、nDCG@10 0.2846、HitRate@10 0.765。5,000 Chunk Pilot：Dense 的 Recall/MRR/nDCG/HitRate 分别为 0.0076/1.0000/0.9863/1.0000；Hybrid 为 0.0069/0.9700/0.9013/1.0000。代理 qrels 不等价于人工类案相关性。

## 15. Reranker 指标

同一 Pilot 中，Hybrid + Qwen3-Reranker-0.6B 为 Recall@10 0.0072、MRR@10 0.9800、nDCG@10 0.9467、HitRate@10 1.0000。它改善了 Hybrid 的 MRR/nDCG，但仍低于 Dense；没有据此声称完成了法律相关性提升。

## 16. 生成评测结果

Prompt Smoke（20 条）：JSON 可解析率 0、Schema 通过率 0、罪名字段 Micro-F1 0。模型输出使用了中文字段名，暴露了严格 Schema 遵循问题。

## 17. SFT 前后对比

生成 SFT Smoke 使用 100/20 条，训练 46.6 秒、端到端 58.3 秒、峰值 9.46 GiB。SFT 后同一 20 条仍为 JSON 可解析率 0、Schema 通过率 0、罪名字段 Micro-F1 0；小样本 Smoke 没有带来可验证改善，因此不包装为有效优化。

## 18. Bad Case 和改进

分类样例真实暴露“抢夺/盗窃”混淆；流水线通过标签阈值检测将其标记为 `uncertain_below_threshold`，只把 Top-3 候选用于分析并触发人工复核。法条检索改为罪名候选辅助，每个候选最多取一个法条，同时保留最多一个事实检索补充条款。生成 Bad Case 主要是字段语言和输出长度失控。

## 19. DPO/RL 决策

暂不实施。SFT 格式尚未稳定，训练数据缺少可核验法条标签，检索评测仍使用代理 qrels，也没有人工审核 chosen/rejected 对。详情见 `reports/alignment/rl_decision.md`。满足门槛后优先小规模 DPO，不优先 GRPO。

## 20. pytest 和 Ruff

最终验收命令结果：`44 passed in 4.17s`；`ruff check .` 通过；`ruff format --check .` 显示 93 files already formatted。

## 21. 新增、修改和备份文件

本轮新增或重构 111 个项目文件，完整列表由 `/tmp/legalmind_changed_files.txt` 生成。主要分组：

- `src/legalmind/data/`：合同、清洗、泄漏、去重、切分、统计、构建与校验
- `src/legalmind/training/`：动态 Padding、长度分桶、长文本、SFT Mask 与生成训练
- `src/legalmind/retrieval/`：Chunk、BM25、Embedding、Hybrid、Reranker、难负样本
- `src/legalmind/generation/`：Schema、证据约束输出、SFT 数据与校验
- `src/legalmind/pipeline/`：统一推理、降级、不确定性和引用校验
- `configs/`、`scripts/`、`tests/`、`reports/`、`docs/interview/`

备份：`/root/autodl-tmp/backups/legalmind_full_build_20260817T105521Z/project_code_before_build.tar.gz`，SHA256 `06908bdfcf6666acbfa9228bf07ee341ffb715ec4ba28b6f7f734a772ef31ee3`。

## 22. 数据构建命令

```bash
OMP_NUM_THREADS=8 PYTHONPATH=src python -m legalmind.data.build_dataset \
  --config configs/data/processed_v2.yaml
```

输出目录存在时命令会拒绝覆盖；复现实验应使用新的输出目录或先保留现有产物。

## 23. 训练命令

```bash
OMP_NUM_THREADS=8 PYTHONPATH=src python scripts/train_qlora.py \
  --training-config configs/classification/qwen3_4b_qlora_full.yaml
```

## 24. 评测命令

```bash
OMP_NUM_THREADS=8 PYTHONPATH=src python -m legalmind.evaluation.evaluate_classification \
  --predictions <predictions.jsonl> --references data/processed_v2/test.jsonl
```

最终 test 只允许在模型和阈值确认后执行一次。

## 25. 推理命令

```bash
OMP_NUM_THREADS=8 PYTHONPATH=src python -m legalmind.pipeline.analyze_case \
  --fact-file examples/case.txt --config configs/pipeline/default.yaml \
  --output artifacts/experiments/pipeline_demo_v2_final.json
```

该命令实际运行总耗时 0.7143 秒，引用校验通过，并因分类阈值不足触发人工复核。

## 26. 未完成或待测事项

- v2 Full QLoRA 正在运行，7,526 steps；完成后才可填写验证 Micro-F1
- 只有确认验证配置后才运行 v2 test
- 三种长文本策略的 Full 消融未运行
- 人工类案 qrels、全量难负样本 Reranker 训练未完成
- 生成 Full SFT 未运行；当前 Smoke 明确失败
- 法条解析和数据来源仍需人工/官方来源复核

## 27. 已知局限

CAIL 历史派生文件缺少 `relevant_articles`，限制了法条监督和生成 SFT 的引用质量；数据标签可能有噪声；共享罪名代理相关性无法替代法律专家类案判断；系统输出只用于算法实验，不构成法律意见。

## 28. 中文简历项目经历

可直接使用的版本位于 `docs/interview/resume_zh.md`。在 v2 Full 完成前，简历指标只允许使用标明来源的历史 10K 结果或 TF-IDF 验证结果，不得写入 v2 Full 计划值。

## 29. 英文简历项目经历

可直接使用的版本位于 `docs/interview/resume_en.md`，指标约束同上。

## 30. 推荐 Git commit message

```text
feat: build evidence-grounded LegalMind-RAG evaluation pipeline
```

当前远程目录没有 `.git`，只有 `.source-commit=1daafc5`；以上为提交建议，尚未创建提交。
