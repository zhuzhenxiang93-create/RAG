# 意图路由端到端评测计划

## 研究问题

1. LoRA 是否优于 Lite 关键词基线？
2. 意图路由是否改善检索，而不只是提高分类分数？
3. 低置信度回退能否减少错误路由造成的召回损失？

## 实验矩阵

| 实验 | 变量 | 固定项 | 指标 |
|---|---|---|---|
| 分类基线 | Lite / LoRA | MASSIVE test | Accuracy、Macro-F1、ECE |
| LoRA 结构 | r=8/16/32 | 数据、seed、epoch | Macro-F1、参数量、显存 |
| 注入层 | q/v 与 q/k/v/o | r、数据 | Macro-F1、训练时间 |
| 语言 | zh-CN 与 zh-CN+en-US | 模型、超参 | 中文/英文分组 F1 |
| 路由消融 | 无路由 / gold / predicted | 同一知识库 | Recall@5、MRR、nDCG@5 |
| 阈值 | 0.4/0.5/0.6/0.7 | 同一 checkpoint | 路由覆盖率、错误路由率、Recall@5 |

## 防止数据泄漏

- 训练只使用 MASSIVE train，选模只使用 validation，最终一次性报告 test；
- 端到端 RAG 路由集另行编写，不能把 Lite 关键词表直接复制成测试问题；
- 保存 dataset revision、标准化 SHA-256、随机种子、参数和逐条预测；
- 调过阈值的数据集称为 calibration，不再称为 test。

## 当前状态

脚本和指标已经实现，GPU 全量实验尚未运行。所有模型指标和“提升百分比”均为待测。

