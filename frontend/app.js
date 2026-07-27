const API = "/api";
const state = { documents: [], health: null, evaluation: null };

const titles = {
  overview: "文档智能工作台",
  documents: "企业文档库",
  qa: "可信问答与证据",
  evaluation: "实验与评测中心",
};

const demoQueries = [
  "SEV-1 生产事故要求多少分钟内响应？",
  "安全策略编号 SEC-2025-07 对生产密钥有什么要求？",
  "RRF 为什么不直接相加 BM25 和 Dense 的原始分数？",
  "比较 2024 和 2025 制度中用户操作日志保留期限的差异。",
  "量子发动机的最大推力是多少？",
];

function escapeHtml(value = "") {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;",
  })[char]);
}

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, options);
  if (!response.ok) {
    let message = `请求失败（${response.status}）`;
    try {
      const payload = await response.json();
      message = payload?.error?.message || payload?.detail?.[0]?.msg || message;
    } catch (_) { /* response may not be JSON */ }
    throw new Error(message);
  }
  return response.status === 204 ? null : response.json();
}

function toast(message, type = "success") {
  const node = document.createElement("div");
  node.className = `toast ${type}`;
  node.textContent = message;
  document.querySelector("#toast-region").appendChild(node);
  setTimeout(() => node.remove(), 3600);
}

function setBusy(button, busy, busyText = "处理中…") {
  if (busy) {
    button.dataset.label = button.textContent;
    button.textContent = busyText;
    button.disabled = true;
  } else {
    button.textContent = button.dataset.label || button.textContent;
    button.disabled = false;
  }
}

function showView(name) {
  document.querySelectorAll(".view").forEach((view) => view.classList.toggle("active", view.id === `view-${name}`));
  document.querySelectorAll(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === name));
  document.querySelector("#page-title").textContent = titles[name];
  history.replaceState(null, "", `#${name}`);
}

async function loadHealth() {
  try {
    state.health = await request("/health");
    document.querySelector(".pulse").classList.add("online");
    document.querySelector("#system-status").textContent = "服务运行正常";
    document.querySelector("#system-meta").textContent = `v${state.health.version} · ${state.health.mode.toUpperCase()}`;
    document.querySelector("#mode-badge").textContent = state.health.mode.toUpperCase();
    document.querySelector("#metric-mode").textContent = state.health.mode.toUpperCase();
    const names = { pdf: "PDF", docx: "Word", xlsx: "Excel", faiss: "FAISS", transformers: "模型推理", ocr: "OCR", llm: "LLM" };
    document.querySelector("#capability-list").innerHTML = Object.entries(state.health.capabilities)
      .map(([key, value]) => `<div class="capability"><span>${names[key] || key}</span><b>${value ? "可用" : "未启用"}</b></div>`).join("");
  } catch (error) {
    document.querySelector("#system-status").textContent = "服务连接失败";
    document.querySelector("#system-meta").textContent = error.message;
    toast(error.message, "error");
  }
}

async function loadDocuments() {
  try {
    const payload = await request("/documents");
    state.documents = payload.documents;
    renderDocuments();
    document.querySelector("#metric-documents").textContent = payload.total;
    document.querySelector("#metric-chunks").textContent = payload.documents.reduce((sum, doc) => sum + doc.child_chunk_count, 0);
    document.querySelector("#document-count-label").textContent = payload.total;
  } catch (error) {
    toast(error.message, "error");
  }
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

function renderDocuments(filter = "") {
  const rows = state.documents.filter((doc) => doc.filename.toLowerCase().includes(filter.toLowerCase()));
  const table = document.querySelector("#document-table");
  if (!rows.length) {
    table.innerHTML = `<tr><td colspan="6"><div class="empty-state">${state.documents.length ? "没有匹配的文档" : "文档库为空，可上传文档或载入演示语料"}</div></td></tr>`;
    return;
  }
  table.innerHTML = rows.map((doc) => `
    <tr>
      <td><div class="file-cell"><span class="file-icon">${escapeHtml(doc.file_type.toUpperCase())}</span><div><strong>${escapeHtml(doc.filename)}</strong><small>${new Date(doc.created_at).toLocaleString("zh-CN")}</small></div></div></td>
      <td>${escapeHtml(doc.file_type.toUpperCase())}</td>
      <td>${doc.parent_chunk_count} 父块 · ${doc.child_chunk_count} 子块</td>
      <td>${formatBytes(doc.size_bytes)}</td>
      <td><span class="status-pill">${doc.status === "ready" ? "可检索" : "失败"}</span></td>
      <td><button class="row-action" data-detail="${doc.document_id}">查看 →</button></td>
    </tr>`).join("");
}

async function bootstrapDemo() {
  const button = document.querySelector("#bootstrap-button");
  setBusy(button, true, "正在初始化…");
  try {
    const payload = await request("/demo/bootstrap", { method: "POST" });
    const message = payload.added.length
      ? `已导入 ${payload.added.length} 篇演示文档，并完成索引`
      : `演示语料已存在，索引已刷新`;
    toast(message);
    await loadDocuments();
  } catch (error) {
    toast(error.message, "error");
  } finally {
    setBusy(button, false);
  }
}

async function uploadFiles(files) {
  const queue = document.querySelector("#upload-queue");
  for (const file of files) {
    const row = document.createElement("div");
    row.className = "upload-row";
    row.innerHTML = `<span>${escapeHtml(file.name)}</span><strong>解析中…</strong>`;
    queue.prepend(row);
    const body = new FormData();
    body.append("file", file);
    try {
      await request("/documents/upload", { method: "POST", body });
      row.querySelector("strong").textContent = "完成";
    } catch (error) {
      row.querySelector("strong").textContent = error.message;
      row.style.color = "var(--red)";
    }
  }
  await rebuildIndex(false);
  await loadDocuments();
}

async function rebuildIndex(notify = true) {
  const button = document.querySelector("#rebuild-button");
  setBusy(button, true, "构建中…");
  try {
    const payload = await request("/index/build", { method: "POST" });
    if (notify) toast(`索引已重建：${payload.document_count} 篇文档，${payload.child_chunk_count} 个子块`);
    return payload;
  } catch (error) {
    toast(error.message, "error");
  } finally {
    setBusy(button, false);
  }
}

async function showDocument(documentId) {
  try {
    const doc = await request(`/documents/${documentId}`);
    document.querySelector("#dialog-title").textContent = doc.filename;
    document.querySelector("#dialog-body").innerHTML = `
      <div class="detail-stats">
        <div class="detail-stat"><span>解析元素</span><strong>${doc.element_count}</strong></div>
        <div class="detail-stat"><span>父级块</span><strong>${doc.parent_chunk_count}</strong></div>
        <div class="detail-stat"><span>子级块</span><strong>${doc.child_chunk_count}</strong></div>
      </div>
      ${doc.chunks.filter((chunk) => chunk.level === "child").slice(0, 5).map((chunk) => `
        <div class="chunk-preview"><b>${escapeHtml(chunk.section || "未命名章节")}${chunk.page ? ` · 第 ${chunk.page} 页` : ""}</b><p>${escapeHtml(chunk.content.slice(0, 300))}</p></div>
      `).join("")}`;
    document.querySelector("#document-dialog").showModal();
  } catch (error) {
    toast(error.message, "error");
  }
}

function renderConfidence(data) {
  const labels = {
    evidence_relevance: "证据相关性",
    channel_agreement: "多路一致性",
    ranking_margin: "排名间隔",
    citation_coverage: "引用覆盖率",
    conflict_penalty: "冲突惩罚",
  };
  document.querySelector("#score-ring").textContent = `${Math.round(data.confidence * 100)}%`;
  document.querySelector("#confidence-list").innerHTML = Object.entries(data.confidence_breakdown).map(([key, value]) => `
    <div class="confidence-row ${key === "conflict_penalty" ? "penalty" : ""}">
      <span>${labels[key]}</span><b>${(value * 100).toFixed(0)}%</b>
      <div class="bar"><i style="width:${Math.max(2, value * 100)}%"></i></div>
    </div>`).join("");
}

function renderRoute(retrieval) {
  const plan = retrieval.plan;
  const intent = plan.intent
    ? `<div class="route-weights">
        <div><span>业务领域</span><b>${escapeHtml(plan.intent_domain || "—")}</b></div>
        <div><span>用户意图</span><b>${escapeHtml(plan.intent)}</b></div>
        <div><span>路由置信度</span><b>${((plan.intent_confidence || 0) * 100).toFixed(0)}%</b></div>
        <div><span>知识库建议</span><b>${escapeHtml(plan.knowledge_base || "全部文档")}</b></div>
      </div>`
    : "";
  document.querySelector("#route-content").innerHTML = `
    <span class="route-type">${escapeHtml(plan.query_type.toUpperCase())}</span>
    ${intent}
    <div class="route-weights">
      <div><span>BM25 权重</span><b>${plan.bm25_weight.toFixed(2)}</b></div>
      <div><span>Dense 权重</span><b>${plan.dense_weight.toFixed(2)}</b></div>
      <div><span>启用重排</span><b>${plan.use_reranker ? "是" : "否"}</b></div>
      <div><span>候选数量</span><b>${retrieval.total_candidates}</b></div>
    </div>
    <p class="route-reason">${escapeHtml(plan.reason)} · ${plan.routing_abstained ? "低置信度回退全库 · " : ""}总耗时 ${Object.values(retrieval.timings_ms).reduce((a, b) => a + b, 0).toFixed(2)} ms</p>`;
}

function renderAnswer(data) {
  const panel = document.querySelector("#answer-panel");
  panel.classList.remove("empty-answer");
  panel.querySelector(".answer-placeholder").hidden = true;
  panel.querySelector(".answer-content").hidden = false;
  const badge = document.querySelector("#decision-badge");
  badge.textContent = data.decision.toUpperCase();
  badge.className = `decision-badge ${data.decision}`;
  document.querySelector("#answer-confidence").textContent = `置信度 ${(data.confidence * 100).toFixed(1)}%`;
  document.querySelector("#answer-text").textContent = data.answer;
  document.querySelector("#answer-reason").textContent = data.reason;

  const evidencePanel = document.querySelector("#evidence-panel");
  evidencePanel.hidden = !data.citations.length;
  document.querySelector("#citation-count").textContent = `${data.citations.length} 条证据`;
  document.querySelector("#evidence-list").innerHTML = data.citations.map((item) => `
    <div class="evidence-item">
      <div class="evidence-top"><strong>[${escapeHtml(item.evidence_id)}] ${escapeHtml(item.filename)}</strong><span><i class="relation ${item.relation}">${escapeHtml(item.relation)}</i> · 相关性 ${(item.relevance * 100).toFixed(0)}%</span></div>
      <p>${escapeHtml(item.text)}</p>
    </div>`).join("");

  const conflictPanel = document.querySelector("#conflict-panel");
  conflictPanel.hidden = !data.conflicts.length;
  document.querySelector("#conflict-list").innerHTML = data.conflicts.map((item) => `
    <div class="conflict-item"><strong>${escapeHtml(item.conflict_type.toUpperCase())}</strong><br>${escapeHtml(item.description)}</div>`).join("");
  renderConfidence(data);
  renderRoute(data.retrieval);
}

async function askQuestion() {
  const input = document.querySelector("#query-input");
  const button = document.querySelector("#ask-button");
  const query = input.value.trim();
  if (!query) return toast("请先输入问题", "error");
  button.textContent = "…";
  button.disabled = true;
  try {
    const payload = await request("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, strategy: document.querySelector("#strategy-select").value, top_k: 5 }),
    });
    renderAnswer(payload);
  } catch (error) {
    toast(`${error.message}。请先载入语料并构建索引。`, "error");
  } finally {
    button.textContent = "→";
    button.disabled = false;
  }
}

function renderEvaluation(data) {
  state.evaluation = data;
  document.querySelector("#eval-run-id").textContent = `RUN ${data.run_id}`;
  const entries = Object.entries(data.retrieval_metrics);
  document.querySelector("#retrieval-chart").className = "bar-chart";
  document.querySelector("#retrieval-chart").innerHTML = entries.map(([name, metrics]) => `
    <div class="chart-column ${metrics.mrr === Math.max(...entries.map(([, item]) => item.mrr)) ? "best" : ""}">
      <span class="value">${metrics.mrr.toFixed(3)}</span>
      <i class="column" style="height:${Math.max(4, metrics.mrr * 190)}px"></i>
      <span class="label">${escapeHtml(name.replace("_", " + "))}</span>
    </div>`).join("");
  const qa = data.qa_metrics;
  const tiles = [
    ["决策准确率", qa.decision_accuracy],
    ["拒答精确率", qa.abstention_precision],
    ["冲突召回率", qa.conflict_recall],
    ["引用召回率", qa.citation_recall],
  ];
  document.querySelector("#qa-metrics").innerHTML = tiles.map(([label, value]) => `
    <div class="metric-tile"><span>${label}</span><strong>${(value * 100).toFixed(1)}%</strong></div>`).join("");
}

async function runEvaluation() {
  const button = document.querySelector("#evaluation-button");
  setBusy(button, true, "正在运行完整评测…");
  try {
    const payload = await request("/evaluation/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ benchmark: "lite_v1" }),
    });
    renderEvaluation(payload);
    toast(`评测完成：${payload.question_count} 个问题`);
  } catch (error) {
    toast(error.message, "error");
  } finally {
    setBusy(button, false);
  }
}

function seedPrompts() {
  document.querySelector("#overview-prompts").innerHTML = demoQueries.slice(0, 3).map((query) =>
    `<button class="prompt-item" data-query="${escapeHtml(query)}">${escapeHtml(query)}<span>→</span></button>`).join("");
  document.querySelector("#query-examples").innerHTML = demoQueries.map((query, index) =>
    `<button class="query-chip" data-query="${escapeHtml(query)}">示例 ${index + 1}</button>`).join("");
}

function bindEvents() {
  document.querySelectorAll(".nav-item").forEach((button) => button.addEventListener("click", () => showView(button.dataset.view)));
  document.querySelectorAll("[data-go]").forEach((button) => button.addEventListener("click", () => showView(button.dataset.go)));
  document.addEventListener("click", (event) => {
    const prompt = event.target.closest("[data-query]");
    if (prompt) {
      document.querySelector("#query-input").value = prompt.dataset.query;
      showView("qa");
    }
    const detail = event.target.closest("[data-detail]");
    if (detail) showDocument(detail.dataset.detail);
  });
  document.querySelector("#bootstrap-button").addEventListener("click", bootstrapDemo);
  document.querySelector("#rebuild-button").addEventListener("click", () => rebuildIndex());
  document.querySelector("#ask-button").addEventListener("click", askQuestion);
  document.querySelector("#query-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) askQuestion();
  });
  document.querySelector("#evaluation-button").addEventListener("click", runEvaluation);
  document.querySelector("#document-filter").addEventListener("input", (event) => renderDocuments(event.target.value));
  document.querySelector("#file-input").addEventListener("change", (event) => uploadFiles([...event.target.files]));
  const dropzone = document.querySelector("#dropzone");
  ["dragenter", "dragover"].forEach((name) => dropzone.addEventListener(name, (event) => { event.preventDefault(); dropzone.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((name) => dropzone.addEventListener(name, (event) => { event.preventDefault(); dropzone.classList.remove("dragging"); }));
  dropzone.addEventListener("drop", (event) => uploadFiles([...event.dataTransfer.files]));
  document.querySelector(".dialog-close").addEventListener("click", () => document.querySelector("#document-dialog").close());
}

async function init() {
  seedPrompts();
  bindEvents();
  const initialView = location.hash.slice(1);
  if (titles[initialView]) showView(initialView);
  await Promise.all([loadHealth(), loadDocuments()]);
}

init();
