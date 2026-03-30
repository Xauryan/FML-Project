from dataclasses import dataclass
import argparse
import csv
import os
import sys

import matplotlib
import numpy as np
import seaborn as sns
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix

if os.environ.get("DISPLAY") is None and sys.platform not in {"win32", "darwin"}:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.fonts import configure_matplotlib_chinese
from src.modeling import (
    DEFAULT_ACTIVATIONS,
    build_lenet5,
    load_dataset,
    set_random_seed,
)
from src.paths import MODEL_PATH, OUTPUT_DIR, ensure_project_dirs

ensure_project_dirs()
sns.set_style("whitegrid")
configure_matplotlib_chinese()

PALETTE = sns.color_palette("Set2", 6)
DPI = 200
BATCH_SIZE = 32
EPOCHS = 20


@dataclass(frozen=True)
class TrainingResult:
    activation: str
    history: dict
    validation_loss: float
    validation_accuracy: float
    test_loss: float
    test_accuracy: float
    saved_path: str


def _save_or_show(fig, filename, show):
    fig.savefig(OUTPUT_DIR / filename, dpi=DPI, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_training_curves(
    history, title_suffix="", filename="training_curves.png", show=True
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor("#F6FAFD")
    epochs_range = range(1, len(history.history["loss"]) + 1)

    ax1.plot(
        epochs_range,
        history.history["loss"],
        "o-",
        color=PALETTE[0],
        label="训练损失",
        linewidth=2,
        markersize=5,
    )
    ax1.plot(
        epochs_range,
        history.history["val_loss"],
        "s--",
        color=PALETTE[1],
        label="验证损失",
        linewidth=2,
        markersize=5,
    )
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title(f"损失曲线{title_suffix}", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(
        epochs_range,
        history.history["accuracy"],
        "o-",
        color=PALETTE[2],
        label="训练准确率",
        linewidth=2,
        markersize=5,
    )
    ax2.plot(
        epochs_range,
        history.history["val_accuracy"],
        "s--",
        color=PALETTE[3],
        label="验证准确率",
        linewidth=2,
        markersize=5,
    )
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title(f"准确率曲线{title_suffix}", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_or_show(fig, filename, show)


def plot_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png", show=True):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#F6FAFD")
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=range(10),
        yticklabels=range(10),
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 11},
    )
    ax.set_xlabel("预测标签", fontsize=13)
    ax.set_ylabel("真实标签", fontsize=13)
    ax.set_title("混淆矩阵", fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save_or_show(fig, filename, show)


def plot_activation_comparison(results, filename="activation_comparison.png", show=True):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#F6FAFD")
    colors = {"tanh": PALETTE[0], "relu": PALETTE[1], "sigmoid": PALETTE[2]}
    markers = {"tanh": "o", "relu": "s", "sigmoid": "^"}

    for activation, result in results.items():
        epochs_range = range(1, len(result.history["val_accuracy"]) + 1)
        ax1.plot(
            epochs_range,
            result.history["val_accuracy"],
            f"{markers[activation]}-",
            color=colors[activation],
            label=activation,
            linewidth=2,
            markersize=5,
        )
        ax2.plot(
            epochs_range,
            result.history["val_loss"],
            f"{markers[activation]}--",
            color=colors[activation],
            label=activation,
            linewidth=2,
            markersize=5,
        )

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("验证准确率", fontsize=12)
    ax1.set_title("激活函数对比 - 验证准确率", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=12, title="激活函数")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("验证损失", fontsize=12)
    ax2.set_title("激活函数对比 - 验证损失", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=12, title="激活函数")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_or_show(fig, filename, show)


def train_with_activation(activation, dataset, show_plots=True):
    model = build_lenet5(activation=activation)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ),
    ]
    history = model.fit(
        dataset.train.images,
        dataset.train.labels_cat,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_data=(dataset.validation.images, dataset.validation.labels_cat),
        callbacks=callbacks,
    )

    validation_loss, validation_accuracy = model.evaluate(
        dataset.validation.images,
        dataset.validation.labels_cat,
        verbose=0,
    )
    test_loss, test_accuracy = model.evaluate(
        dataset.test.images,
        dataset.test.labels_cat,
        verbose=0,
    )
    saved_path = OUTPUT_DIR / f"lenet5_{activation}.keras"
    model.save(saved_path)
    plot_training_curves(
        history,
        f" ({activation})",
        f"training_curves_{activation}.png",
        show=show_plots,
    )

    return model, TrainingResult(
        activation=activation,
        history=history.history,
        validation_loss=float(validation_loss),
        validation_accuracy=float(validation_accuracy),
        test_loss=float(test_loss),
        test_accuracy=float(test_accuracy),
        saved_path=str(saved_path),
    )


def save_model_summary(model):
    summary_path = OUTPUT_DIR / "model_summary.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        model.summary(print_fn=lambda line: handle.write(line + "\n"))


def save_activation_results(results):
    csv_path = OUTPUT_DIR / "activation_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "activation",
                "validation_accuracy",
                "validation_loss",
                "test_accuracy",
                "test_loss",
                "saved_path",
            ]
        )
        for activation, result in results.items():
            writer.writerow(
                [
                    activation,
                    f"{result.validation_accuracy:.4f}",
                    f"{result.validation_loss:.4f}",
                    f"{result.test_accuracy:.4f}",
                    f"{result.test_loss:.4f}",
                    result.saved_path,
                ]
            )


def parse_args():
    parser = argparse.ArgumentParser(description="训练 LeNet-5 并输出评估图表。")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="只保存图像，不弹出 matplotlib 窗口。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed()
    dataset = load_dataset()
    print(
        "数据集划分:",
        f"train={dataset.train.images.shape}",
        f"val={dataset.validation.images.shape}",
        f"test={dataset.test.images.shape}",
    )

    results = {}
    best_activation = None
    best_result = None
    best_model = None

    for activation in DEFAULT_ACTIVATIONS:
        print(f"\n{'=' * 50}")
        print(f"训练 LeNet-5 (激活函数: {activation})")
        print(f"{'=' * 50}")

        model, result = train_with_activation(
            activation,
            dataset,
            show_plots=not args.no_show,
        )
        print(
            f"[{activation}] 验证损失: {result.validation_loss:.4f}, "
            f"验证准确率: {result.validation_accuracy:.4f}, "
            f"测试损失: {result.test_loss:.4f}, "
            f"测试准确率: {result.test_accuracy:.4f}"
        )
        results[activation] = result

        if (
            best_result is None
            or result.validation_accuracy > best_result.validation_accuracy
        ):
            best_activation = activation
            best_result = result
            best_model = model

    plot_activation_comparison(results, show=not args.no_show)

    print(f"\n{'=' * 50}")
    print("激活函数对比结果汇总")
    print(f"{'=' * 50}")
    print(f"{'激活函数':<12} {'测试准确率':<15} {'测试损失':<15}")
    print("-" * 42)
    for activation, result in results.items():
        print(
            f"{activation:<12} {result.test_accuracy:<15.4f} {result.test_loss:<15.4f}"
        )
    print(
        f"\n最佳: {best_activation} (验证准确率: {best_result.validation_accuracy:.4f})"
    )
    print(
        f"最佳模型测试损失: {best_result.test_loss:.4f}, "
        f"测试准确率: {best_result.test_accuracy:.4f}"
    )

    predictions = np.argmax(best_model.predict(dataset.test.images, verbose=0), axis=1)
    plot_confusion_matrix(
        dataset.test.labels,
        predictions,
        show=not args.no_show,
    )

    best_model.save(MODEL_PATH)
    print(f"最佳模型已保存为 {MODEL_PATH.name}")

    save_activation_results(results)
    save_model_summary(best_model)
    best_model.summary()


if __name__ == "__main__":
    main()
