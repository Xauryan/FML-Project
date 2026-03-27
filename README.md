# FML-Project

MNIST handwriting recognition based on LeNet-5, with training, visualization, and interactive GUI.

## Stack

| Category | Tools |
|----------|-------|
| ML | Python 3.12 / Keras 3 / TensorFlow |
| Visualization | Matplotlib / Seaborn / scikit-learn |
| GUI | Tkinter + Tauri (ONNX Runtime Web) |

## Quick Start

```bash
git clone https://github.com/Xauryan/FML-Project

cd FML-Project

conda env create -f environment.yml

conda activate fml

python -m src.train       # train + activation comparison

python -m src.visualize   # architecture / filters / feature maps

python -m src.gui         # tkinter gui
```

### Tauri App (Optional)

```bash
python scripts/keras_to_onnx.py
cd tauri-app && pnpm install && pnpm tauri dev
```

Cross-platform installers (macOS / Windows / Linux) are built automatically via GitHub Actions on merge to `main`, and published to [Releases](https://github.com/Xauryan/FML-Project/releases).

## Structure

```
├── src/
│   ├── modeling.py
│   ├── train.py
│   ├── visualize.py
│   ├── gui.py
│   ├── inference.py
│   ├── preprocessing.py
│   ├── fonts.py
│   └── paths.py
├── scripts/
│   └── keras_to_onnx.py
├── tauri-app/
├── tests/
├── models/
└── outputs/
```
