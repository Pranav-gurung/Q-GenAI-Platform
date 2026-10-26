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
    for (const k of ["order_id", "operator_id"]) if (form[k].value.trim()) payload[k] = form[k].value.trim();
    const created = await api("/inspections", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    render(created);
    const data = new FormData();
    data.append("file", form.image.files[0]);
    render(await api(`/inspections/${created.inspection_id}/image`, { method: "POST", body: data }), form.image.files[0]);
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
