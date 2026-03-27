import { existsSync, mkdirSync, readFileSync, statSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import process from "node:process";

const appDir = resolve(import.meta.dirname, "..");
const repoRoot = resolve(appDir, "..");
const modelPath = resolve(repoRoot, "models", "mnist.keras");
const converterPath = resolve(repoRoot, "scripts", "keras_to_onnx.py");
const outputPath = resolve(appDir, "public", "model.onnx");
const environmentPath = resolve(repoRoot, "environment.yml");

function isUpToDate() {
  if (!existsSync(outputPath)) {
    return false;
  }

  const outputMtime = statSync(outputPath).mtimeMs;
  return [modelPath, converterPath].every((path) => statSync(path).mtimeMs <= outputMtime);
}

function discoverPython() {
  const candidates = [];
  const activeCondaPrefix = process.env.CONDA_PREFIX;
  if (activeCondaPrefix) {
    const condaPython =
      process.platform === "win32"
        ? resolve(activeCondaPrefix, "python.exe")
        : resolve(activeCondaPrefix, "bin", "python");
    if (existsSync(condaPython)) {
      candidates.push([condaPython]);
    }
  }

  if (existsSync(environmentPath)) {
    const envFile = readFileSync(environmentPath, "utf8");
    const match = envFile.match(/^name:\s*([^\s]+)\s*$/m);
    if (match) {
      candidates.push(["conda", "run", "--no-capture-output", "-n", match[1], "python"]);
    }
  }

  if (process.env.PYTHON) {
    candidates.push([process.env.PYTHON]);
  }

  if (process.platform === "win32") {
    candidates.push(["python"], ["py", "-3"]);
  } else {
    candidates.push(["python3"], ["python"]);
  }

  for (const candidate of candidates) {
    const probe = spawnSync(
      candidate[0],
      [
        ...candidate.slice(1),
        "-c",
        "import numpy, tensorflow, keras, tf2onnx, onnxruntime",
      ],
      {
        encoding: "utf8",
      },
    );

    if (probe.status === 0) {
      return candidate;
    }
  }

  throw new Error("未找到可用的 Python 环境，请先安装并配置项目依赖。");
}

if (!existsSync(modelPath)) {
  throw new Error(`缺少模型文件: ${modelPath}`);
}

mkdirSync(dirname(outputPath), { recursive: true });

if (isUpToDate()) {
  console.log(`model.onnx 已是最新，跳过转换: ${outputPath}`);
  process.exit(0);
}

const python = discoverPython();
const result = spawnSync(
  python[0],
  [...python.slice(1), converterPath, modelPath, outputPath],
  {
    cwd: repoRoot,
    stdio: "inherit",
  },
);

if (result.status !== 0) {
  process.exit(result.status ?? 1);
}
