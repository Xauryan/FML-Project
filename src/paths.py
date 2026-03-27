from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
FONTS_DIR = ROOT_DIR / "fonts"
TAURI_APP_DIR = ROOT_DIR / "tauri-app"
TAURI_PUBLIC_DIR = TAURI_APP_DIR / "public"
TAURI_ICONS_DIR = TAURI_APP_DIR / "src-tauri" / "icons"

MODEL_PATH = MODELS_DIR / "mnist.keras"
TAURI_MODEL_PATH = TAURI_PUBLIC_DIR / "model.onnx"
BUNDLED_FONT_PATH = FONTS_DIR / "SourceHanSansSC-Regular.otf"
APP_ICON_SVG_PATH = TAURI_APP_DIR / "app-icon.svg"
APP_ICON_PNG_PATH = TAURI_ICONS_DIR / "icon.png"


def ensure_project_dirs():
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    TAURI_PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    TAURI_ICONS_DIR.mkdir(parents=True, exist_ok=True)
