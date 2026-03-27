import * as ort from "onnxruntime-web";
import ortWasmJsepUrl from "onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm?url";
import { preprocessCanvasImageData } from "./digit-preprocessing.js";

let session = null;

ort.env.wasm.wasmPaths = { wasm: ortWasmJsepUrl };

async function loadModel() {
  try {
    session = await ort.InferenceSession.create("/model.onnx");
    console.log("model loaded");
  } catch (err) {
    console.error("model load failed:", err);
    document.getElementById("result-conf").textContent = "模型加载失败";
  }
}

async function predict(imageData, width, height) {
  if (!session) return null;

  const input = preprocessCanvasImageData(imageData, width, height);
  if (!input) return null;

  const tensor = new ort.Tensor("float32", input, [1, 28, 28, 1]);
  const results = await session.run({ input: tensor });
  const output = results[Object.keys(results)[0]];
  return Array.from(output.data);
}

const canvas = document.getElementById("draw-canvas");
const ctx = canvas.getContext("2d");
let isDrawing = false;
let lastX = 0;
let lastY = 0;

function clearCanvas() {
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  document.getElementById("result-digit").textContent = "?";
  document.getElementById("result-conf").textContent = "在左侧画布书写数字";
  renderBars(null);
}

function startDraw(x, y) {
  isDrawing = true;
  lastX = x;
  lastY = y;
}

function draw(x, y) {
  if (!isDrawing) return;
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(x, y);
  ctx.strokeStyle = "#000";
  ctx.lineWidth = 16;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.stroke();
  lastX = x;
  lastY = y;
}

function stopDraw() {
  isDrawing = false;
}

canvas.addEventListener("mousedown", (e) => startDraw(e.offsetX, e.offsetY));
canvas.addEventListener("mousemove", (e) => draw(e.offsetX, e.offsetY));
canvas.addEventListener("mouseup", stopDraw);
canvas.addEventListener("mouseleave", stopDraw);

canvas.addEventListener("touchstart", (e) => {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const t = e.touches[0];
  startDraw(t.clientX - rect.left, t.clientY - rect.top);
});
canvas.addEventListener("touchmove", (e) => {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const t = e.touches[0];
  draw(t.clientX - rect.left, t.clientY - rect.top);
});
canvas.addEventListener("touchend", (e) => {
  e.preventDefault();
  stopDraw();
});

function initBars() {
  const container = document.getElementById("prob-bars");
  container.innerHTML = "";
  for (let i = 0; i < 10; i++) {
    container.innerHTML += `
      <div class="prob-row">
        <span class="prob-label">${i}</span>
        <div class="prob-bar-bg">
          <div class="prob-bar-fill" id="bar-${i}" style="width: 0%"></div>
        </div>
        <span class="prob-value" id="val-${i}">0%</span>
      </div>`;
  }
}

function renderBars(probs) {
  const topIdx = probs ? probs.indexOf(Math.max(...probs)) : -1;
  for (let i = 0; i < 10; i++) {
    const bar = document.getElementById(`bar-${i}`);
    const val = document.getElementById(`val-${i}`);
    if (!bar || !val) {
      continue;
    }
    const pct = probs ? (probs[i] * 100).toFixed(1) : 0;
    bar.style.width = `${pct}%`;
    bar.className = "prob-bar-fill" + (i === topIdx ? " top" : "");
    val.textContent = probs ? `${pct}%` : "0%";
  }
}

document.getElementById("btn-recognize").addEventListener("click", async () => {
  if (!session) {
    document.getElementById("result-conf").textContent = "模型未加载";
    return;
  }
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  const probs = await predict(imageData, canvas.width, canvas.height);
  if (!probs) {
    document.getElementById("result-conf").textContent = "请先书写数字";
    return;
  }

  const digit = probs.indexOf(Math.max(...probs));
  const conf = (Math.max(...probs) * 100).toFixed(1);
  document.getElementById("result-digit").textContent = digit;
  document.getElementById("result-conf").textContent = `置信度: ${conf}%`;
  renderBars(probs);
});

document.getElementById("btn-clear").addEventListener("click", clearCanvas);

initBars();
clearCanvas();
loadModel();
