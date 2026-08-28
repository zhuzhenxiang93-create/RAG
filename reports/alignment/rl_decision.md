# DPO / RL 决策记录

状态：`not_run_threshold_not_met`

## 已完成的前置工作

- Qwen3-4B 生成 QLoRA Smoke 已完成，规模为 100 条训练、20 条验证。
- 旧 Smoke 已逐条区分 JSON 解析、Schema 匹配和截断错误；SFT v3.1 数据已构造并通过自动校验。
- Assistant-only loss、Chat Template、EOS、动态 Padding、Checkpoint 与 Adapter 保存链路已验证。
- 结构化输出定义了 JSON Schema，并实现了引用存在性校验。
- 法条与案例检索已经有 BM25 基线；Dense 与预训练 Reranker 处于 Pilot 评测阶段。

## 当前不进入 DPO 的依据

1. 生成 SFT 目前只有 20 条 Smoke 诊断；规范化修复后的 Schema 通过率为 0.95，但 v3.1 Pilot 尚未训练，稳定性仍未证明。
2. 当前迁移的旧 CAIL 派生数据缺失 `relevant_articles`，无法构造经过验证的法条偏好对。
3. 检索评测的相关性定义仍是“共享罪名”的自动代理标签，尚无人工 qrels。
4. 没有经过人工审核的 chosen/rejected 数据；自动生成记录均保持 `unreviewed`。
5. 当前错误仍可优先通过补齐原始数据、检索评测、SFT 数据审核和规则校验解决。

## 重新评估门槛

满足以下条件后再考虑小规模 DPO：

- 生成 SFT 在独立验证集上获得稳定的 JSON 可解析率和 Schema 通过率；
- 法条与案例引用的来源字段完整，且引用正确率可自动验证；
- 至少完成一轮生成 Bad Case 人工审核，形成可重复的偏好错误类型；
- chosen/rejected 只来自训练集，并通过事实、引用和 split 泄漏校验；
- DPO 目标无法通过数据修复、Prompt、检索或普通 SFT 以更小成本解决。

第一轮若满足门槛，优先选择 DPO Pilot，不加入 GRPO。计划奖励规则只作为校验设计，当前没有声称任何 RL 指标。
