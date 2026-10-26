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

function render(insp) {
  document.getElementById("result").hidden = false;
  document.getElementById("r-id").textContent = insp.inspection_id;
  const status = document.getElementById("r-status");
  status.textContent = insp.status;
  status.className = "badge s-" + (["COMPLETED", "REQUIRES_REVIEW", "FAILED"].includes(insp.status) ? insp.status : "other");
  document.getElementById("r-decision").textContent = insp.decision ?? "None (no automated decision)";
  const last = [...(insp.history ?? [])].reverse().find(h => h.event_type === "STATE_TRANSITION");
  document.getElementById("r-reason").textContent = insp.error?.message ?? last?.message ?? "-";
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
    render(await api(`/inspections/${created.inspection_id}/image`, { method: "POST", body: data }));
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
