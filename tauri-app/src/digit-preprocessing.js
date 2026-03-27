const MODEL_INPUT_SIZE = 28;
const CONTENT_SIZE = 20;
const INK_THRESHOLD = 240;

function createCanvas(width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

function findInkBounds(imageData, width, height) {
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const offset = (y * width + x) * 4;
      const gray =
        (imageData[offset] + imageData[offset + 1] + imageData[offset + 2]) / 3;
      if (gray < INK_THRESHOLD) {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    }
  }

  if (maxX === -1) {
    return null;
  }

  return { minX, minY, maxX, maxY };
}

export function preprocessCanvasImageData(imageData, width, height) {
  const bounds = findInkBounds(imageData, width, height);
  if (!bounds) {
    return null;
  }

  const srcCanvas = createCanvas(width, height);
  srcCanvas
    .getContext("2d")
    .putImageData(new ImageData(new Uint8ClampedArray(imageData), width, height), 0, 0);

  const cropWidth = bounds.maxX - bounds.minX + 1;
  const cropHeight = bounds.maxY - bounds.minY + 1;
  const scale = CONTENT_SIZE / Math.max(cropWidth, cropHeight);
  const resizedWidth = Math.max(1, Math.round(cropWidth * scale));
  const resizedHeight = Math.max(1, Math.round(cropHeight * scale));

  const normalizedCanvas = createCanvas(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
  const normalizedCtx = normalizedCanvas.getContext("2d");
  normalizedCtx.fillStyle = "#fff";
  normalizedCtx.fillRect(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
  normalizedCtx.imageSmoothingEnabled = true;
  normalizedCtx.drawImage(
    srcCanvas,
    bounds.minX,
    bounds.minY,
    cropWidth,
    cropHeight,
    Math.floor((MODEL_INPUT_SIZE - resizedWidth) / 2),
    Math.floor((MODEL_INPUT_SIZE - resizedHeight) / 2),
    resizedWidth,
    resizedHeight,
  );

  const normalized = normalizedCtx.getImageData(
    0,
    0,
    MODEL_INPUT_SIZE,
    MODEL_INPUT_SIZE,
  ).data;
  const input = new Float32Array(MODEL_INPUT_SIZE * MODEL_INPUT_SIZE);
  let hasInk = false;

  for (let index = 0; index < MODEL_INPUT_SIZE * MODEL_INPUT_SIZE; index += 1) {
    const offset = index * 4;
    const gray =
      (normalized[offset] + normalized[offset + 1] + normalized[offset + 2]) / 3;
    const value = (255 - gray) / 255;
    input[index] = value;
    if (value > 0.01) {
      hasInk = true;
    }
  }

  return hasInk ? input : null;
}
