const POLL_INTERVAL_MS = 1000;
const POLL_TIMEOUT_MS = 10 * 60 * 1000;
let previewImage = null;
let datasetId = null;
let datasetPreview = null;

function status(id, message) { document.querySelector(id).textContent = message; }
function fileLabel(input) { const label = input.closest("label").querySelector(".file-name"); if (input.files[0]) label.textContent = `${input.files[0].name} · ${(input.files[0].size / 1024 / 1024).toFixed(2)} MB`; }
async function pollJob(jobId, statusId) {
  const deadline = Date.now() + POLL_TIMEOUT_MS;
  for (;;) {
    const response = await fetch(`/jobs/${jobId}`); const job = await response.json();
    if (!response.ok) throw new Error(job.detail || "Could not read job status.");
    status(statusId, `Job ${jobId}: ${job.status}`);
    if (job.status === "finished") return job.result;
    if (["failed", "stopped", "canceled"].includes(job.status)) throw new Error(job.error || `Job ended with status: ${job.status}.`);
    if (Date.now() >= deadline) throw new Error("Timed out waiting for the job.");
    await new Promise(resolve => setTimeout(resolve, POLL_INTERVAL_MS));
  }
}
function drawDetections(detections) {
  const canvas = document.querySelector("#canvas"), context = canvas.getContext("2d");
  canvas.width = previewImage.naturalWidth; canvas.height = previewImage.naturalHeight; context.drawImage(previewImage, 0, 0);
  context.lineWidth = Math.max(2, canvas.width / 350); context.font = `${Math.max(14, canvas.width / 45)}px sans-serif`; context.textBaseline = "top";
  const list = document.querySelector("#detections"); list.replaceChildren();
  for (const item of detections) { const [x1, y1, x2, y2] = item.box, caption = `${item.label} ${(item.score * 100).toFixed(1)}%`;
    context.strokeStyle = "#e72d21"; context.strokeRect(x1, y1, x2 - x1, y2 - y1); const width = context.measureText(caption).width + 10, height = Math.max(20, canvas.width / 28);
    context.fillStyle = "#0067b5"; context.fillRect(x1, Math.max(0, y1 - height), width, height); context.fillStyle = "white"; context.fillText(caption, x1 + 5, Math.max(0, y1 - height + 2));
    const li = document.createElement("li"); li.textContent = caption; list.append(li);
  }
  document.querySelector("#detection-count").textContent = `${detections.length} object${detections.length === 1 ? "" : "s"} detected`;
}
function bindImageForm(formId, inputId, endpoint, statusId, done) {
  const form = document.querySelector(formId), input = document.querySelector(inputId), button = form.querySelector("button");
  input.addEventListener("change", () => fileLabel(input));
  form.addEventListener("submit", async event => { event.preventDefault(); const file = input.files[0]; if (!file) return; button.disabled = true;
    try { if (endpoint === "/jobs") { previewImage = new Image(); const url = URL.createObjectURL(file); previewImage.src = url; await previewImage.decode(); URL.revokeObjectURL(url); }
      const body = new FormData(); body.append("image", file); status(statusId, "Uploading image and creating a job…"); const response = await fetch(endpoint, {method: "POST", body}); const job = await response.json(); if (!response.ok) throw new Error(job.detail || "Could not create job."); const result = await pollJob(job.job_id, statusId); done(result); status(statusId, "Analysis complete.");
    } catch (error) { status(statusId, `Error: ${error.message}`); } finally { button.disabled = false; }
  });
}
bindImageForm("#detection-form", "#detection-image", "/jobs", "#detection-status", result => { drawDetections(result.detections); document.querySelector("#detection-result").hidden = false; });
bindImageForm("#classification-form", "#classification-image", "/classification/jobs", "#classification-status", result => { const list = document.querySelector("#classification-predictions"); list.replaceChildren(); result.predictions.forEach(item => { const li = document.createElement("li"); li.textContent = `${item.label} — ${(item.score * 100).toFixed(1)}%`; list.append(li); }); document.querySelector("#classification-result").hidden = false; });

function renderPreview(data) {
  datasetPreview = data; const sheet = document.querySelector("#sheet-select"), target = document.querySelector("#target-select"); sheet.replaceChildren(); target.replaceChildren();
  data.sheet_names.forEach(name => { const option = new Option(name, name, false, name === data.selected_sheet); sheet.add(option); });
  data.columns.forEach(column => target.add(new Option(`${column.name} (${column.dtype})`, column.name)));
  document.querySelector("#dataset-summary").textContent = `${data.rows} rows · ${data.columns.length} columns · selected sheet: ${data.selected_sheet}`;
  const table = document.createElement("table"), head = document.createElement("tr"); data.columns.forEach(column => { const th = document.createElement("th"); th.textContent = column.name; head.append(th); }); table.append(head);
  data.preview.forEach(row => { const tr = document.createElement("tr"); data.columns.forEach(column => { const td = document.createElement("td"); td.textContent = row[column.name] ?? ""; tr.append(td); }); table.append(tr); });
  document.querySelector("#dataset-table").replaceChildren(table); document.querySelector("#training-panel").hidden = false;
}
async function refreshSheet() { if (!datasetId) return; const response = await fetch(`/tabular/datasets/${datasetId}/preview?sheet=${encodeURIComponent(document.querySelector("#sheet-select").value)}`); const data = await response.json(); if (!response.ok) throw new Error(data.detail); renderPreview(data); }
document.querySelector("#dataset-file").addEventListener("change", event => fileLabel(event.target));
document.querySelector("#dataset-form").addEventListener("submit", async event => { event.preventDefault(); const input = document.querySelector("#dataset-file"), file = input.files[0]; if (!file) return; const button = event.currentTarget.querySelector("button"); button.disabled = true;
  try { const body = new FormData(); body.append("dataset", file); const response = await fetch("/tabular/datasets", {method: "POST", body}); const data = await response.json(); if (!response.ok) throw new Error(data.detail); datasetId = data.dataset_id; renderPreview(data); status("#training-status", "Choose a target parameter and training type."); } catch (error) { status("#training-status", `Error: ${error.message}`); } finally { button.disabled = false; }
});
document.querySelector("#sheet-select").addEventListener("change", () => refreshSheet().catch(error => status("#training-status", `Error: ${error.message}`)));
document.querySelector("#train-button").addEventListener("click", async () => { if (!datasetId) return; const button = document.querySelector("#train-button"); button.disabled = true; try { const response = await fetch("/tabular/training-jobs", {method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify({dataset_id: datasetId, sheet: document.querySelector("#sheet-select").value, target: document.querySelector("#target-select").value, task: document.querySelector("#task-select").value})}); const job = await response.json(); if (!response.ok) throw new Error(job.detail); status("#training-status", "Training ANN…"); await pollJob(job.job_id, "#training-status"); status("#training-status", "Training complete. Latest ANN model saved."); await loadLatestModel(); } catch (error) { status("#training-status", `Error: ${error.message}`); } finally { button.disabled = false; } });
async function loadLatestModel() { const response = await fetch("/tabular/model"); if (response.status === 404) return; const model = await response.json(); if (!response.ok) throw new Error(model.detail); const metrics = Object.entries(model.metrics).map(([name, value]) => `${name}: ${value}`).join(", "); document.querySelector("#model-summary").textContent = `${model.task} model predicting “${model.target}” from ${model.source_filename}, sheet ${model.sheet}. Trained ${new Date(model.trained_at).toLocaleString()}. Features: ${model.feature_columns.join(", ")}. ${metrics}.`; document.querySelector("#prediction-description").textContent = `Enter the ${model.feature_columns.length} numeric parameters used to train ${model.target}.`; const fields = document.querySelector("#prediction-fields"); fields.replaceChildren(); model.feature_columns.forEach(name => { const label = document.createElement("label"); label.textContent = name; const input = document.createElement("input"); input.type = "number"; input.step = "any"; input.name = name; input.required = true; label.append(input); fields.append(label); }); document.querySelector("#prediction-form").hidden = false; }
document.querySelector("#prediction-form").addEventListener("submit", async event => { event.preventDefault(); const features = Object.fromEntries(new FormData(event.currentTarget)); try { const response = await fetch("/tabular/predictions", {method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify({features})}); const result = await response.json(); if (!response.ok) throw new Error(result.detail); document.querySelector("#prediction-result").textContent = result.task === "regression" ? `${result.target}: ${result.prediction}` : `${result.target}: ${result.prediction} (${Object.entries(result.probabilities).map(([name, value]) => `${name} ${(value * 100).toFixed(1)}%`).join(", ")})`; } catch (error) { document.querySelector("#prediction-result").textContent = `Error: ${error.message}`; } });
loadLatestModel().catch(error => status("#training-status", `Error loading latest model: ${error.message}`));

let cocoCursors = [null], cocoPage = 0, cocoSelected = null, cocoBoxesVisible = true;
const cocoBrowser = document.querySelector("#coco-browser");
function drawCocoBoxes() {
  const image = document.querySelector("#coco-image"), canvas = document.querySelector("#coco-canvas");
  if (!cocoSelected || !cocoBoxesVisible || !image.naturalWidth) { canvas.hidden = true; return; }
  canvas.hidden = false; canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
  const context = canvas.getContext("2d"); context.clearRect(0, 0, canvas.width, canvas.height); context.lineWidth = Math.max(2, canvas.width / 350); context.font = `${Math.max(14, canvas.width / 42)}px sans-serif`; context.textBaseline = "top";
  cocoSelected.annotations.forEach(annotation => { const [x, y, width, height] = annotation.bbox, label = annotation.label;
    context.strokeStyle = "#e72d21"; context.strokeRect(x, y, width, height); const labelWidth = context.measureText(label).width + 10, labelHeight = Math.max(20, canvas.width / 28);
    context.fillStyle = "#0067b5"; context.fillRect(x, Math.max(0, y - labelHeight), labelWidth, labelHeight); context.fillStyle = "white"; context.fillText(label, x + 5, Math.max(0, y - labelHeight + 2));
  });
}
async function selectCocoImage(key) {
  const response = await fetch(`/coco/images/${encodeURIComponent(key)}`), data = await response.json(); if (!response.ok) throw new Error(data.detail);
  cocoSelected = data; const image = document.querySelector("#coco-image"); image.src = data.url; image.onload = drawCocoBoxes;
  document.querySelector("#coco-image-title").textContent = data.filename;
  document.querySelector("#coco-image-meta").textContent = `${data.width} × ${data.height} · ${data.annotation_count} annotations`;
  const labels = document.querySelector("#coco-labels"); labels.replaceChildren(); data.annotations.forEach(annotation => { const li = document.createElement("li"); li.textContent = annotation.label; labels.append(li); });
  document.querySelector("#coco-detail").hidden = false; document.querySelector("#coco-toggle-boxes").textContent = cocoBoxesVisible ? "Hide bounding boxes" : "Show bounding boxes";
}
async function loadCocoPage(page = 0) {
  const split = document.querySelector("#coco-split").value, category = document.querySelector("#coco-category").value, cursor = cocoCursors[page];
  const params = new URLSearchParams({split, limit: "24"}); if (category) params.set("category_id", category); if (cursor) params.set("cursor", cursor);
  const response = await fetch(`/coco/images?${params}`), data = await response.json(); if (!response.ok) throw new Error(data.detail);
  cocoPage = page; cocoCursors = cocoCursors.slice(0, page + 1); if (data.next_cursor) cocoCursors[page + 1] = data.next_cursor;
  const grid = document.querySelector("#coco-thumbnails"); grid.replaceChildren(); data.images.forEach(item => { const button = document.createElement("button"); button.type = "button"; button.className = "coco-thumbnail"; const image = document.createElement("img"); image.src = item.url; image.alt = item.filename; const caption = document.createElement("span"); caption.textContent = `${item.split} · ${item.annotation_count} labels`; button.append(image, caption); button.addEventListener("click", () => selectCocoImage(item.image_key).catch(error => status("#coco-status", `Error: ${error.message}`))); grid.append(button); });
  document.querySelector("#coco-previous").disabled = page === 0; document.querySelector("#coco-next").disabled = !data.next_cursor;
}
async function initialiseCoco() {
  const response = await fetch("/coco/status"), data = await response.json(); if (!response.ok) throw new Error(data.detail);
  const splitSummary = Object.entries(data.splits || {}).map(([name, value]) => `${name}: ${value.images || 0} images`).join(", ");
  status("#coco-status", data.status === "ready" ? `COCO 2017 ready. ${splitSummary}` : `COCO import status: ${data.status}. ${data.error || "Run the documented importer command to load the dataset."}`);
  if (data.status !== "ready") return;
  const category = document.querySelector("#coco-category"); data.categories.forEach(item => category.add(new Option(item.name, item.category_id)));
  cocoBrowser.hidden = false; await loadCocoPage();
}
document.querySelector("#coco-load").addEventListener("click", () => { cocoCursors = [null]; loadCocoPage().catch(error => status("#coco-status", `Error: ${error.message}`)); });
document.querySelector("#coco-previous").addEventListener("click", () => loadCocoPage(cocoPage - 1).catch(error => status("#coco-status", `Error: ${error.message}`)));
document.querySelector("#coco-next").addEventListener("click", () => loadCocoPage(cocoPage + 1).catch(error => status("#coco-status", `Error: ${error.message}`)));
document.querySelector("#coco-toggle-boxes").addEventListener("click", () => { cocoBoxesVisible = !cocoBoxesVisible; document.querySelector("#coco-toggle-boxes").textContent = cocoBoxesVisible ? "Hide bounding boxes" : "Show bounding boxes"; drawCocoBoxes(); });
initialiseCoco().catch(error => status("#coco-status", `COCO dataset unavailable: ${error.message}`));
