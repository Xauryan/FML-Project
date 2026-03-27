import argparse
import os
import sys

import matplotlib
import numpy as np
import seaborn as sns

import keras
from keras.datasets import mnist
from keras.models import Model

if os.environ.get("DISPLAY") is None and sys.platform not in {"win32", "darwin"}:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.fonts import configure_matplotlib_chinese
from src.inference import load_digit_model
from src.paths import MODEL_PATH, OUTPUT_DIR, ensure_project_dirs

ensure_project_dirs()
sns.set_style("whitegrid")
configure_matplotlib_chinese()
DPI = 200


def save_or_show(fig, save_path, show):
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"图像已保存: {save_path}")


def visualize_architecture(model, show=True):
    arch_path = OUTPUT_DIR / "model_architecture.png"
    try:
        keras.utils.plot_model(
            model,
            to_file=str(arch_path),
            show_shapes=True,
            show_layer_names=True,
            show_layer_activations=True,
            dpi=DPI,
        )
        print(f"架构图已保存: {arch_path}")
    except Exception as error:
        print(f"plot_model 失败 ({error})，使用 matplotlib 绘制")
        _plot_architecture_fallback(model, arch_path, show)


def _plot_architecture_fallback(model, save_path, show=True):
    layers_info = []
    for layer in model.layers:
        output_shape = getattr(layer, "output_shape", layer.output.shape)
        params = layer.count_params()
        layers_info.append((layer.name, layer.__class__.__name__, output_shape, params))

    fig, ax = plt.subplots(figsize=(7.4, max(len(layers_info) * 1.25, 6.4)))
    fig.patch.set_facecolor("#F6FAFD")
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, len(layers_info) + 0.5)
    ax.axis("off")

    box_colors = {
        "Conv2D": "#4ECDC4",
        "AveragePooling2D": "#45B7D1",
        "Flatten": "#96CEB4",
        "Dense": "#FFEAA7",
    }

    for index, (name, layer_type, shape, params) in enumerate(reversed(layers_info)):
        y = index
        rect = plt.Rectangle(
            (1, y - 0.35),
            8,
            0.7,
            facecolor=box_colors.get(layer_type, "#DFE6E9"),
            edgecolor="#2D3436",
            linewidth=1.5,
            zorder=2,
            alpha=0.85,
            joinstyle="round",
        )
        ax.add_patch(rect)

        shape_str = str(shape).replace("None", "batch")
        label = f"{name} ({layer_type})\n输出: {shape_str}  参数: {params:,}"
        ax.text(
            5,
            y,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            zorder=3,
        )

        if index < len(layers_info) - 1:
            ax.annotate(
                "",
                xy=(5, y + 0.35),
                xytext=(5, y + 0.65),
                arrowprops=dict(arrowstyle="->", color="#636E72", lw=1.5),
            )

    total_params = sum(params for _, _, _, params in layers_info)
    ax.set_title(
        f"LeNet-5 架构 (总参数: {total_params:,})",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    fig.tight_layout(pad=1.2)
    save_or_show(fig, save_path, show)


def visualize_filters(model, show=True):
    conv_layers = [
        layer for layer in model.layers if isinstance(layer, keras.layers.Conv2D)
    ]

    for layer in conv_layers:
        weights = layer.get_weights()[0]
        n_filters = weights.shape[-1]
        n_cols = min(n_filters, 8)
        n_rows = (n_filters + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * 1.8, max(n_rows * 1.8, 3.8)),
        )
        fig.patch.set_facecolor("#F6FAFD")
        axes = np.atleast_2d(axes)

        fig.suptitle(
            f"卷积核权重 - {layer.name} ({n_filters}个滤波器, "
            f"{weights.shape[0]}x{weights.shape[1]})",
            fontsize=13,
            fontweight="bold",
        )

        vmin, vmax = weights.min(), weights.max()
        image = None
        for row in range(n_rows):
            for col in range(n_cols):
                index = row * n_cols + col
                ax = axes[row][col]
                if index < n_filters:
                    kernel = weights[:, :, 0, index]
                    image = ax.imshow(
                        kernel,
                        cmap="coolwarm",
                        vmin=vmin,
                        vmax=vmax,
                        interpolation="nearest",
                    )
                    ax.set_title(f"F{index}", fontsize=9)
                ax.axis("off")

        fig.subplots_adjust(top=0.88, right=0.78)
        if image is not None:
            cbar_ax = fig.add_axes([0.82, 0.15, 0.03, 0.6])
            fig.colorbar(image, cax=cbar_ax, label="权重值")
        save_path = OUTPUT_DIR / f"filters_{layer.name}.png"
        save_or_show(fig, save_path, show)


def visualize_feature_maps(model, image, show=True):
    target_layers = [
        layer
        for layer in model.layers
        if isinstance(
            layer,
            (
                keras.layers.Conv2D,
                keras.layers.AveragePooling2D,
                keras.layers.MaxPooling2D,
            ),
        )
    ]

    feature_model = Model(
        inputs=model.inputs, outputs=[layer.output for layer in target_layers]
    )
    feature_maps = feature_model.predict(image)
    if not isinstance(feature_maps, list):
        feature_maps = [feature_maps]

    fig_in, ax_in = plt.subplots(figsize=(3.5, 3.5))
    fig_in.patch.set_facecolor("#F6FAFD")
    ax_in.imshow(image.reshape(28, 28), cmap="gray")
    ax_in.set_title("输入图像", fontsize=14, fontweight="bold")
    ax_in.axis("off")
    fig_in.tight_layout()
    save_or_show(fig_in, OUTPUT_DIR / "input_image.png", show)

    for layer, fmap in zip(target_layers, feature_maps):
        n_channels = fmap.shape[-1]
        n_cols = min(n_channels, 8)
        n_rows = (n_channels + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * 2, max(n_rows * 2, 3.8)),
        )
        fig.patch.set_facecolor("#F6FAFD")
        axes = np.atleast_2d(axes)

        fig.suptitle(
            f"特征图 - {layer.name} ({layer.__class__.__name__}, "
            f"{n_channels}通道, {fmap.shape[1]}x{fmap.shape[2]})",
            fontsize=13,
            fontweight="bold",
        )

        for row in range(n_rows):
            for col in range(n_cols):
                index = row * n_cols + col
                ax = axes[row][col]
                if index < n_channels:
                    ax.imshow(fmap[0, :, :, index], cmap="viridis")
                    ax.set_title(f"Ch{index}", fontsize=9)
                ax.axis("off")

        fig.subplots_adjust(top=0.88)
        save_path = OUTPUT_DIR / f"feature_map_{layer.name}.png"
        save_or_show(fig, save_path, show)


def parse_args():
    parser = argparse.ArgumentParser(description="生成模型结构与特征可视化图。")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="只保存图像，不弹出 matplotlib 窗口。",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=3,
        help="用于特征图可视化的测试集样本索引。",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not MODEL_PATH.exists():
        print(f"模型文件不存在: {MODEL_PATH}")
        print("请先运行 `python -m src.train`")
        return

    model = load_digit_model()
    model.summary()

    print("\n[1/3] 架构图")
    visualize_architecture(model, show=not args.no_show)

    print("\n[2/3] 卷积核权重")
    visualize_filters(model, show=not args.no_show)

    print("\n[3/3] 特征图")
    (_, _), (x_test, _) = mnist.load_data()
    sample_index = max(0, min(args.sample_index, len(x_test) - 1))
    sample = x_test[sample_index].reshape(1, 28, 28, 1).astype("float32") / 255.0
    visualize_feature_maps(model, sample, show=not args.no_show)

    print(f"\n完成，输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
