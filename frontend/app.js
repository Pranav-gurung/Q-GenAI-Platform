// Minimal Part 1 operator UI: create inspection -> upload image -> show pipeline status.
const API = "/api";
const form = document.getElementById("inspection-form");
const errorBox = document.getElementById("error");
const preview = document.getElementById("preview");
const STAGE_LABELS = { QUALITY_CHECK: "Image quality", CV_PROCESSING: "Computer vision", DECISION: "Decision",
  PIPELINE_ANALYSIS: "Fulfillment analysis", RCA_PROCESSING: "Root-cause analysis", ACTION_GENERATED: "Corrective action" };

async function api(path, options = {}) {
  const res = await fetch(API + path, options);
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const err = body?.error ?? { code: "HTTP_" + res.status, message: res.statusText };
    const details = Array.isArray(err.details) ? "\n" + err.details.map(d => `- ${d.field}: ${d.message}`).join("\n") : "";
    throw new Error(`${err.code}: ${err.message}${details}`);
  }
  return body;
}

function fill(select, items, value, label) {
  select.replaceChildren(...items.map(i => new Option(label(i), value(i))));
}

function showError(msg) { errorBox.textContent = msg; errorBox.hidden = !msg; }

const SEV_COLORS = { LOW: "#2ea043", MEDIUM: "#dbab09", HIGH: "#e66414", CRITICAL: "#d71e28" };

// Draws the boxes returned by the real CV engine onto the image the operator uploaded.
// Boxes are pixel coordinates of the EXIF-oriented image; browsers apply EXIF orientation too.
async function drawDetections(file, detections) {
  const canvas = document.getElementById("r-canvas");
  if (!file) { canvas.hidden = true; return; }
  const bmp = await createImageBitmap(file);
  canvas.width = bmp.width; canvas.height = bmp.height; canvas.hidden = false;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bmp, 0, 0);
  const lw = Math.max(2, Math.round(Math.max(bmp.width, bmp.height) / 300));
  ctx.font = `${Math.max(12, Math.round(Math.max(bmp.width, bmp.height) / 45))}px system-ui, sans-serif`;
  for (const d of detections) {
    if (!d.bbox) continue;
    const { x_min, y_min, x_max, y_max } = d.bbox;
    const color = SEV_COLORS[d.severity] ?? "#7878ff";
    ctx.lineWidth = lw; ctx.strokeStyle = color; ctx.strokeRect(x_min, y_min, x_max - x_min, y_max - y_min);
    const text = `${d.defect_type} ${d.confidence.toFixed(2)}`;
    const h = parseInt(ctx.font, 10) + 6, w = ctx.measureText(text).width + 8;
    const ty = y_min > h ? y_min - h : y_max;
    ctx.fillStyle = color; ctx.fillRect(x_min, ty, w, h);
    ctx.fillStyle = "#fff"; ctx.fillText(text, x_min + 4, ty + h - 5);
  }
}

function renderCv(insp, file) {
  const box = document.getElementById("r-cv");
  const cv = insp.result?.cv;
  if (!cv) { box.hidden = true; return; }
  box.hidden = false;
  const meta = document.getElementById("r-cv-meta");
  const rows = document.querySelector("#r-detections tbody");
  rows.replaceChildren();
  if (cv.status !== "COMPLETED") {
    meta.textContent = `${cv.module}: ${cv.status}${cv.message ? " (" + cv.message + ")" : ""}`;
    document.getElementById("r-canvas").hidden = true;
    document.getElementById("r-detections").hidden = true;
    return;
  }
  const inf = cv.inference;
  meta.textContent = `Model ${inf.model_name}:${inf.model_version} on ${inf.inference_device} · ` +
    `${inf.inference_time_ms.toFixed(0)} ms inference · threshold ${inf.confidence_threshold} · ` +
    `${cv.detections.length} detection(s)`;
  document.getElementById("r-detections").hidden = cv.detections.length === 0;
  for (const d of cv.detections) {
    const tr = document.createElement("tr");
    const b = d.bbox;
    for (const [text, cls] of [[d.defect_type], [d.confidence.toFixed(2)], [d.severity ?? "-", "sev-" + d.severity],
        [b ? `${Math.round(b.x_min)},${Math.round(b.y_min)} - ${Math.round(b.x_max)},${Math.round(b.y_max)}` : "-"]]) {
      const td = document.createElement("td"); td.textContent = text; if (cls) td.className = cls; tr.append(td);
    }
    rows.append(tr);
  }
  drawDetections(file, cv.detections).catch(() => { document.getElementById("r-canvas").hidden = true; });
}

const label = (s) => s ? s.charAt(0) + s.slice(1).toLowerCase().replaceAll("_", " ") : "-";

function renderFulfillment(trace) {
  const box = document.getElementById("r-fulfillment");
  if (!trace) { box.hidden = true; return; }
  box.hidden = false;
  const key = trace.trace;
  document.getElementById("r-trace").textContent = key.key_type === "NONE"
    ? "No item or order id: this inspection cannot be linked to other stages."
    : `${key.key_type === "ITEM" ? "Item " + key.item_id : "Order " + key.order_id} · ${key.sku_id} · ` +
      `${trace.observations.length} observation(s) · ${label(trace.status)}`;
  document.getElementById("r-timeline").replaceChildren(...trace.timeline.map(t => {
    const li = document.createElement("li");
    li.className = "tl-" + t.status + (t.stage === trace.current_stage ? " current" : "");
    const b = document.createElement("b"); b.textContent = t.stage;
    li.append(b, document.createTextNode(t.status === "NOT_OBSERVED" ? "not observed" : t.status));
    if (t.stage === trace.current_stage) li.style.outline = "2px solid currentColor";
    return li;
  }));
  document.getElementById("r-defects").replaceChildren(...trace.defect_analyses.map(a => {
    const card = document.createElement("div"); card.className = "defect-card";
    const h = document.createElement("strong"); h.textContent = `${a.defect_type} · ${label(a.classification)}`;
    const dl = document.createElement("dl");
    for (const [k, v] of [["First observed", a.first_observed_stage], ["Last known good", a.last_known_good_stage ?? "none"],
        ["Not inspected", a.missing_stages.length ? a.missing_stages.join(", ") : "none"],
        ["Evidence", a.evidence_level], ["This stage", label(a.current_stage_attribution)],
        ["Why", label(a.explanation_code)]]) {
      const dt = document.createElement("dt"); dt.textContent = k;
      const dd = document.createElement("dd"); dd.textContent = v;
      dl.append(dt, dd);
    }
    card.append(h, dl);
    return card;
  }));
}

function listBlock(title, items, cls, fmt = (x) => x) {
  const wrap = document.createElement("div");
  if (!items.length) return wrap;
  const h = document.createElement("h4"); h.textContent = title;
  const ul = document.createElement("ul"); if (cls) ul.className = cls;
  for (const it of items) { const li = document.createElement("li"); li.textContent = fmt(it); ul.append(li); }
  wrap.append(h, ul);
  return wrap;
}

function section(title, cls, ...children) {
  const sec = document.createElement("div"); sec.className = `rca-section ${cls}`;
  const h = document.createElement("h4"); h.textContent = title;
  sec.append(h, ...children);
  return sec;
}

const RAG_TONE = { GROUNDED: "ok", NOT_SUPPORTED: "warn", SOURCES_ONLY: "warn", NO_RELEVANT_SOURCE: "warn",
                   UNAVAILABLE: "bad", NOT_APPLICABLE: "neutral" };

// SOP-grounded actions: each shows its source document, section and version (filled from the index, not the LLM).
function sopActions(actions) {
  const ul = document.createElement("ul"); ul.className = "sop-actions";
  for (const a of actions) {
    const li = document.createElement("li"); li.textContent = `${a.action} (${a.priority})`;
    const src = document.createElement("span"); src.className = "sop-source";
    src.textContent = `Source: ${a.source_document} · Section: ${a.source_section} · Version: ${a.source_version}`;
    li.append(src); ul.append(li);
  }
  return ul;
}

// RCA: system evidence, GenAI hypotheses and SOP-grounded actions are rendered as three separate, labelled sections.
function renderRCA(rca, inspectionId) {
  const box = document.getElementById("r-rca");
  const btn = document.getElementById("r-rca-generate");
  box.hidden = false;
  btn.hidden = false;
  btn.textContent = rca && rca.status === "COMPLETED" ? "Regenerate GenAI RCA" : "Generate GenAI RCA";
  btn.onclick = async () => {
    btn.disabled = true;
    try { renderRCA(await api(`/inspections/${inspectionId}/rca`), inspectionId); }
    catch (err) { showError(err.message); } finally { btn.disabled = false; }
  };
  const meta = document.getElementById("r-rca-meta");
  const defects = document.getElementById("r-rca-defects");
  defects.replaceChildren();
  document.getElementById("r-rca-summary").textContent = rca?.summary ?? "";
  document.getElementById("r-rca-disclaimer").textContent = rca?.disclaimer ?? "";
  renderRAG(rca?.rag);
  if (!rca) { meta.textContent = "No RCA was generated during this inspection."; return; }
  const g = rca.generation;
  meta.textContent = rca.status === "COMPLETED"
    ? `${g.provider === "mock" ? "Deterministic MOCK provider (not AI)" : g.provider + " / " + g.model} · ` +
      `${g.latency_ms != null ? Math.round(g.latency_ms) + " ms" : ""}${rca.requires_review ? " · requires review" : ""}`
    : rca.status === "NOT_APPLICABLE" ? "No defect in this item's history: nothing to explain."
    : `GenAI RCA ${rca.status.toLowerCase().replace("_", " ")} (${g.error_code}). Deterministic results above are ` +
      "unaffected; the inspection requires human review.";
  const rag = rca.rag;
  for (const d of rca.defects) {
    const card = document.createElement("div"); card.className = "rca-block";
    const h = document.createElement("strong");
    h.textContent = `${d.defect_type} · first observed ${d.facts.first_observed_stage} · last known good ` +
      `${d.facts.last_known_good_stage ?? "none"} · evidence ${d.facts.evidence_level}`;
    const unsupported = new Set((rag?.unsupported_actions ?? []).filter(u => u.defect_type === d.defect_type).map(u => u.action));
    const aiActions = rag ? d.corrective_actions.filter(a => unsupported.has(a.action)) : d.corrective_actions;
    card.append(h,
      section("Observed Evidence (system)", "facts",
        listBlock("Facts", d.observed_evidence, "rca-facts"),
        listBlock("Limitations", d.limitations, "rca-facts")),
      section("GenAI Analysis (hypotheses, not facts)", "genai",
        listBlock("Likely contributing causes", d.likely_causes, "rca-ai", c => `${c.hypothesis} [${c.category}]`),
        listBlock(rag ? "AI-suggested actions - not SOP-backed" : "Corrective actions (AI-suggested)", aiActions, "rca-ai",
                  a => `${a.action} (${a.priority})`),
        listBlock("Verification steps", d.verification_steps, "rca-ai"),
        listBlock("Model-stated limitations", d.ai_limitations, "rca-ai")));
    if (rag) {
      const grounded = rag.grounded_actions.filter(a => a.defect_type === d.defect_type);
      const dg = rag.defects.find(x => x.defect_type === d.defect_type);
      const body = grounded.length ? sopActions(grounded) : Object.assign(document.createElement("p"),
        { className: "meta", textContent: dg?.message ?? rag.message ?? "No SOP-backed actions." });
      card.append(section("SOP-Grounded Actions", "sop", body));
    }
    defects.append(card);
  }
}

function renderRAG(rag) {
  const meta = document.getElementById("r-rag-meta");
  const box = document.getElementById("r-rag-sources");
  meta.hidden = box.hidden = !rag;
  if (!rag) return;
  const badge = document.getElementById("r-rag-badge");
  badge.textContent = rag.status.replaceAll("_", " ");
  badge.style.color = `var(--${RAG_TONE[rag.status] ?? "neutral"})`;
  document.getElementById("r-rag-message").textContent =
    (rag.requires_review ? "· manual review recommended " : "") + (rag.message ? `· ${rag.message}` : "");
  const ul = box.querySelector("ul"); ul.replaceChildren();
  for (const s of rag.retrieved_sources) {
    const li = document.createElement("li");
    li.textContent = `${s.title} · ${s.section} · ${s.version} (${s.provenance}) · similarity ${s.similarity.toFixed(2)}`;
    ul.append(li);
  }
  document.getElementById("r-rag-disclaimer").textContent = rag.disclaimer;
}

function render(insp, file) {
  document.getElementById("result").hidden = false;
  document.getElementById("r-id").textContent = insp.inspection_id;
  const status = document.getElementById("r-status");
  status.textContent = insp.status;
  status.className = "badge s-" + (["COMPLETED", "REQUIRES_REVIEW", "FAILED"].includes(insp.status) ? insp.status : "other");
  const decision = insp.result?.decision;
  document.getElementById("r-decision").textContent = insp.decision
    ? `${insp.decision}${decision?.severity ? " · " + decision.severity : ""}${decision?.decision_source ? " (" + decision.decision_source + ")" : ""}`
    : "None (no automated decision)";
  const last = [...(insp.history ?? [])].reverse().find(h => h.event_type === "STATE_TRANSITION");
  document.getElementById("r-reason").textContent =
    insp.error?.message ?? (decision?.status === "COMPLETED" ? decision.reason : null) ?? last?.message ?? "-";
  renderCv(insp, file);
  const stages = (insp.history ?? []).filter(h => h.event_type === "MODULE_RESULT");
  document.getElementById("r-stages").replaceChildren(...stages.map(h => {
    const li = document.createElement("li");
    li.textContent = `${STAGE_LABELS[h.to_state] ?? h.to_state}: ${h.module_status}`;
    const small = document.createElement("small");
    small.textContent = h.message ?? "";
    li.append(small);
    return li;
  }));
}

form.image.addEventListener("change", () => {
  const f = form.image.files[0];
  preview.hidden = !f;
  if (f) preview.src = URL.createObjectURL(f);
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  showError("");
  if (!form.image.files[0]) return showError("Please select an image.");
  const button = form.querySelector("button");
  button.disabled = true;
  try {
    const payload = { store_id: form.store_id.value, sku_id: form.sku_id.value, stage: form.stage.value };
    for (const k of ["item_id", "order_id", "operator_id"]) if (form[k].value.trim()) payload[k] = form[k].value.trim();
    const created = await api("/inspections", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    render(created);
    renderFulfillment(null);
    document.getElementById("r-rca").hidden = true;
    const data = new FormData();
    data.append("file", form.image.files[0]);
    render(await api(`/inspections/${created.inspection_id}/image`, { method: "POST", body: data }), form.image.files[0]);
    renderFulfillment(await api(`/inspections/${created.inspection_id}/pipeline`));
    const done = await api(`/inspections/${created.inspection_id}`);
    const snap = done.result?.rca;
    renderRCA(snap?.root_cause_analysis ?? (snap ? { status: snap.status === "UNAVAILABLE" ? "UNAVAILABLE" : "NOT_APPLICABLE",
      defects: [], generation: { error_code: (snap.message || "").split(":")[0] } } : null), created.inspection_id);
  } catch (err) {
    showError(err.message);
  } finally {
    button.disabled = false;
  }
});

api("/reference-data").then(ref => {
  fill(form.store_id, ref.stores, s => s.store_id, s => `${s.name} (${s.store_id})`);
  fill(form.sku_id, ref.skus, s => s.sku_id, s => `${s.name} (${s.sku_id})`);
  fill(form.stage, ref.stages, s => s.stage, s => s.stage);
  if (!ref.stores.length || !ref.skus.length) showError("No stores/SKUs found. Run: python -m scripts.seed_sample_data");
}).catch(err => showError("Could not load reference data: " + err.message));
