import { existsSync } from "node:fs";
import { resolve } from "node:path";

const target = process.argv[2];

if (!target) {
  console.error("请传入要检查的文件路径。");
  process.exit(1);
}

const resolved = resolve(process.cwd(), target);
if (!existsSync(resolved)) {
  console.error(`缺少构建产物: ${resolved}`);
  process.exit(1);
}

console.log(`文件存在: ${resolved}`);
