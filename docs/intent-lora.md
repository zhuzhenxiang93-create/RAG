# 通用意图 LoRA：数据、训练、推理与评测

## 目标

使用 Qwen2.5-1.5B 序列分类模型和 PEFT LoRA 识别通用服务意图，再把预测结果接入
知识库建议和检索策略。主任务使用 Amazon MASSIVE；项目不分发原始数据或模型权重，
只提供可复现脚本、配置和自编示例。

MASSIVE 官方资源：

- 论文：<https://arxiv.org/abs/2204.08582>
- 仓库：<https://github.com/alexa/massive>
- 数据集：<https://huggingface.co/datasets/AmazonScience/massive>

公开资料说明 MASSIVE 覆盖多语言、18 个领域和 60 个意图。实际下载后的
`manifest.json` 与 `taxonomy.json` 才是本次实验的数据依据。

## 1. 安装 Full 依赖

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-full.txt
```

## 2. 下载并标准化数据

中文单语：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_massive.py `
  --locale zh-CN `
  --revision main `
  --output data\intent\massive
```

中英双语：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_massive.py `
  --locale zh-CN --locale en-US `
  --revision main `
  --output data\intent\massive-bilingual
```

正式复现实验时应把 `--revision` 换成已验证的 Hugging Face commit SHA。脚本保存：

- 标准化后的 train/validation/test JSONL；
- 意图 ID 与领域映射 `taxonomy.json`；
- 数据版本、各 split 数量和标准化内容 SHA-256 的 `manifest.json`。

数据目录已被 Git 忽略。`--limit-per-split 100` 仅用于流水线冒烟，不可作为最终指标。

## 3. 训练 LoRA

```powershell
.\.venv\Scripts\python.exe scripts\train_intent_lora.py `
  --data-dir data\intent\massive `
  --locale zh-CN `
  --base-model Qwen/Qwen2.5-1.5B `
  --output artifacts\intent-lora-zh `
  --rank 16 --alpha 32 `
  --epochs 3 `
  --batch-size 8 `
  --gradient-accumulation-steps 4 `
  --learning-rate 2e-4 `
  --bf16
```

默认注入 `q_proj`、`v_proj`，并通过 `modules_to_save=["score"]` 把分类头和 Adapter
一起保存。训练随机种子默认为 42，验证集 Macro-F1 用于选择最佳 checkpoint。
显卡不支持 BF16 时去掉 `--bf16`，按硬件选择 `--fp16` 或 FP32。

产物：

- `adapter/`：LoRA Adapter、分类头和 tokenizer；
- `taxonomy.json`：标签与领域映射；
- `training_metrics.json`：训练参数和验证指标。

## 4. 独立测试集评测

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_intent.py `
  --data-dir data\intent\massive `
  --locale zh-CN `
  --base-model Qwen/Qwen2.5-1.5B `
  --adapter artifacts\intent-lora-zh\adapter `
  --taxonomy artifacts\intent-lora-zh\taxonomy.json `
  --output artifacts\intent-lora-zh\test-results.json
```

输出 Accuracy、Macro-F1、逐类 Precision/Recall/F1、混淆矩阵、ECE 和逐条预测。
Softmax 概率尚未经过温度缩放，所以 API 的 `calibrated=false`；ECE 只是诊断结果。

## 5. API 使用 LoRA

```dotenv
DOCMIND_INTENT_ENABLED=true
DOCMIND_INTENT_BACKEND=lora
DOCMIND_INTENT_BASE_MODEL=Qwen/Qwen2.5-1.5B
DOCMIND_INTENT_ADAPTER_PATH=D:\LLM\DocMind-RAG\artifacts\intent-lora-zh\adapter
DOCMIND_INTENT_LABELS_PATH=D:\LLM\DocMind-RAG\artifacts\intent-lora-zh\taxonomy.json
DOCMIND_INTENT_DEVICE=auto
DOCMIND_INTENT_CONFIDENCE_THRESHOLD=0.55
```

请求：

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/plugins/intent/classify `
  -H "Content-Type: application/json" `
  -d '{\"text\":\"我的订单为什么还没有配送？\",\"top_k\":3}'
```

当最高概率低于阈值时，路由不会强制选择业务库，而是回退 `general_documents` 和宽召回。
LoRA 加载或推理失败也不会中断基础检索链路。

## 当前已验证与待测

已验证：Lite API、路由集成、资产校验、指标计算和 39 项回归测试。

待测：MASSIVE 全量下载、GPU 训练、LoRA 测试集 Accuracy/Macro-F1/ECE，以及路由前后
RAG Recall@K/MRR 的端到端增益。没有真实产物前不得在简历填写具体提升数字。

