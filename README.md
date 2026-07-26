# DocMind-RAG

复杂企业文档的自适应检索与可信问答系统。

当前版本已完成无密钥可运行的文档接入、混合检索、可信问答、可选法律插件、
可复现评测和面试演示工作台。旧项目保持只读，能力通过明确接口迁移。

## 演示工作台

前端使用原生 HTML/CSS/JavaScript，不依赖 Node 构建链或外部 CDN。提供：

- 文档上传、结构统计、分块预览和索引重建；
- 一键载入 7 篇内置演示语料，按 SHA-256 幂等去重；
- 回答、澄清、拒答三类决策；
- 五维置信度、Query Router、检索权重、证据关系和冲突展示；
- 五种检索策略的实时消融评测看板。

一键启动：

```powershell
.\scripts\run-demo.ps1
```

完整的五分钟讲解顺序和追问回答见 `docs/interview-demo.md`。

## 工程化能力

- `X-Request-ID` 请求追踪与 `Server-Timing` 响应耗时；
- 默认 JSON 结构化访问日志，不记录问题正文、文档内容或密钥；
- `/api/diagnostics` 返回无敏感信息的文档、分块和索引状态；
- 非 root Docker 镜像、健康检查和独立运行数据卷；
- GitHub Actions 执行依赖、语法、测试与镜像构建检查；
- 标准库并发 HTTP 基线脚本，结果保存为 JSON。

部署和运维说明见 `docs/deployment.md`，最终架构见 `docs/architecture.md`。

## 当前支持的文档

- PDF：原生文本提取并保留页码；扫描 PDF 会明确提示启用 OCR
- DOCX：保留标题层级并抽取表格
- XLSX：按工作表生成结构化 Markdown 表格
- TXT
- Markdown：保留标题层级

上传后会生成标题感知的 Parent-Child 分块。Child Chunk 自动携带章节标题，
用于减少旧项目中“子块缺少实体或章节上下文”的问题。

## 检索策略

- `bm25`：适合编号、实体和精确关键词
- `dense`：Lite 模式使用确定性 Hashing 向量作为 CPU 基线
- `rrf`：只融合候选实际出现的排名
- `rrf_rerank`：RRF 后使用透明的 Lite 重排器
- `adaptive`：根据 Query Router 动态设置权重并决定是否重排

搜索响应会返回 BM25/Dense 原始分数、各路排名、RRF 分数、重排分数、
父块上下文、命中通道和阶段耗时。

## 可信回答

`POST /api/chat` 执行：

1. 自适应检索；
2. 证据相关性判断；
3. 跨文档数值、版本和否定冲突检测；
4. 非 LLM 自报的置信度计算；
5. `answer / clarify / abstain` 决策；
6. Lite 模式抽取式回答和证据编号引用。

置信度由证据相关性、多路一致性、排名间隔、引用覆盖率和冲突惩罚组成。
当前冲突检测是可解释规则基线，不应描述为训练后的 NLI 模型。

## 可选法律插件

法律 LoRA 分类已重构为延迟加载插件：

- `GET /api/plugins/legal/status`
- `POST /api/plugins/legal/classify`

默认关闭，不安装 Transformers/PEFT 也不影响主系统。模型、Adapter、分类头和
标签文件均通过环境变量配置，不复制旧项目中的大型权重。详情见
`docs/legal-plugin.md`。

## 可复现评测

```powershell
python -B scripts/evaluate.py --benchmark lite_v1
```

内置 `lite_v1` 是 7 篇人工编写文档和 14 个带金标准标注的问题，用于验证
BM25、Dense、RRF、RRF+Reranker、Adaptive 的消融流程，以及引用、拒答和
冲突检测。它是小型工程基准，不能描述为生产效果或通用模型指标。

接口：

- `POST /api/evaluation/run`
- `GET /api/evaluation/{run_id}`
- `POST /api/demo/bootstrap`（仅 Lite 模式）

详情见 `docs/evaluation.md`。

第一份真实 Lite 基准及失败分析见 `docs/evaluation-results.md`。结果显示该小型
精确查询语料上 BM25 优于 Lite Dense 和 Adaptive，项目不会声称 Adaptive
已经取得提升。

## Lite 模式启动

```powershell
python -m pip install -r requirements-lite.txt
.\scripts\run-dev.ps1
```

打开：

- 健康检查：`http://127.0.0.1:8000/api/health`
- API 文档：`http://127.0.0.1:8000/docs`
- 演示页面：`http://127.0.0.1:8000/ui/`

上传示例：

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/documents/upload `
  -F "file=@data/samples/policy.md"
```

文档接口：

- `POST /api/documents/upload`
- `GET /api/documents`
- `GET /api/documents/{document_id}`
- `DELETE /api/documents/{document_id}`
- `POST /api/index/build`
- `POST /api/search`
- `POST /api/chat`

搜索示例：

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/search `
  -H "Content-Type: application/json" `
  -d '{\"query\":\"编号 ABC-2025 的要求是什么？\",\"strategy\":\"adaptive\",\"top_k\":5}'
```

## 测试

```powershell
.\scripts\test.ps1
```

基础测试使用 Python 标准库 `unittest`，不要求额外测试框架。

轻量并发基线：

```powershell
.\.venv\Scripts\python.exe -B scripts\load_test.py `
  --bootstrap --scenario chat --requests 100 --concurrency 8
```

该结果仅表示单机 Lite 工程基线，不代表生产容量。

## 安全默认值

- 不包含任何真实密钥；
- Lite 模式不连接远程 OCR；
- 导入应用不会加载大模型或创建索引；
- 模型、索引、数据库、上传文件和 `.env` 均被 Git 忽略。

完整技术方案见 `docs/architecture.md` 和阶段 2 冻结文档。

最终交付材料：

- `docs/project-handoff.md`：复现入口、已验证项和待测边界；
- `docs/interview-demo.md`：五分钟演示与追问回答；
- `docs/resume-project-cn.md`：可直接替换的中文简历项目经历；
- `docs/git-commit-plan.md`：Git 初始化和提交拆分建议。
