const form = document.querySelector("#upload-form");
const fileInput = document.querySelector("#image");
const statusText = document.querySelector("#status");
const result = document.querySelector("#result");
const canvas = document.querySelector("#canvas");
const detectionList = document.querySelector("#detections");
const statusPanel = document.querySelector("#status-panel");
const statusDetail = document.querySelector(".status-detail");
const dropzone = document.querySelector("#dropzone");
const fileName = document.querySelector("#file-name");
const submitButton = document.querySelector("#submit-button");
const resultCount = document.querySelector("#result-count");
let previewImage = null;
const POLL_INTERVAL_MS = 1000;
const POLL_TIMEOUT_MS = 10 * 60 * 1000;

function setStatus(message, state = "idle", detail = "Inference queue online") {
  statusText.textContent = message;
  statusPanel.dataset.state = state;
  statusDetail.textContent = detail;
}

function showSelectedFile(file) {
  fileName.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
}

function drawDetections(detections) {
  const context = canvas.getContext("2d");
  canvas.width = previewImage.naturalWidth;
  canvas.height = previewImage.naturalHeight;
  context.drawImage(previewImage, 0, 0);
  context.lineWidth = Math.max(2, canvas.width / 350);
  context.font = `${Math.max(14, canvas.width / 45)}px sans-serif`;
  context.textBaseline = "top";
  detectionList.replaceChildren();
  for (const detection of detections) {
    const [x1, y1, x2, y2] = detection.box;
    const caption = `${detection.label} ${(detection.score * 100).toFixed(1)}%`;
    context.strokeStyle = "#63c5ff";
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const textWidth = context.measureText(caption).width + 10;
    const textHeight = Math.max(20, canvas.width / 28);
    context.fillStyle = "#247fc0";
    context.fillRect(x1, Math.max(0, y1 - textHeight), textWidth, textHeight);
    context.fillStyle = "white";
    context.fillText(caption, x1 + 5, Math.max(0, y1 - textHeight + 2));
    const item = document.createElement("li");
    item.textContent = caption;
    detectionList.append(item);
  }
  resultCount.textContent = `${detections.length} object${detections.length === 1 ? "" : "s"} detected`;
}

async function waitForJob(jobId) {
  const deadline = Date.now() + POLL_TIMEOUT_MS;
  for (;;) {
    const response = await fetch(`/jobs/${jobId}`);
    const job = await response.json();
    if (!response.ok) throw new Error(job.detail || "Could not read job status.");
    setStatus(`Job ${jobId}: ${job.status}`, "working", "Your result is on its way");
    if (job.status === "finished") return job.result;
    if (["failed", "stopped", "canceled"].includes(job.status)) {
      throw new Error(job.error || `Inference ended with status: ${job.status}.`);
    }
    if (Date.now() >= deadline) {
      throw new Error("Timed out waiting for inference. Your job may still be running; refresh to try again.");
    }
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) return;
  result.hidden = true;
  submitButton.disabled = true;
  let objectUrl;
  try {
    previewImage = new Image();
    objectUrl = URL.createObjectURL(file);
    previewImage.src = objectUrl;
    await previewImage.decode();
    setStatus("Uploading image and creating a job…", "working", "Creating a secure queue job");
    const body = new FormData();
    body.append("image", file);
    const response = await fetch("/jobs", { method: "POST", body });
    const job = await response.json();
    if (!response.ok) throw new Error(job.detail || "Could not create job.");
    const output = await waitForJob(job.job_id);
    drawDetections(output.detections);
    result.hidden = false;
    setStatus(`Complete: ${output.detections.length} object(s) at 50% confidence or above.`, "idle", "Analysis complete");
  } catch (error) { setStatus(`Error: ${error.message}`, "error", "Please try another image"); }
  finally { if (objectUrl) URL.revokeObjectURL(objectUrl); submitButton.disabled = false; }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) showSelectedFile(fileInput.files[0]);
});

["dragenter", "dragover"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
  event.preventDefault();
  dropzone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragging");
}));
dropzone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  fileInput.files = transfer.files;
  showSelectedFile(file);
});
