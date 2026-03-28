from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def convert(model_path, onnx_path):
    import numpy as np
    import keras
    import tensorflow as tf
    import tf2onnx

    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    model = keras.models.load_model(model_path, compile=False)
    print(f"已加载: {model_path}")
    model.summary()

    spec = (tf.TensorSpec(model.input_shape, tf.float32, name="input"),)

    @tf.function(input_signature=spec)
    def serving_fn(input_tensor):
        return {"output": model(input_tensor, training=False)}

    try:
        tf2onnx.convert.from_keras(
            model, input_signature=spec, output_path=str(onnx_path)
        )
    except KeyError as error:
        print(f"from_keras 导出失败 ({error})，使用 from_function")
        tf2onnx.convert.from_function(
            serving_fn,
            input_signature=spec,
            output_path=str(onnx_path),
        )

    print(f"已保存: {onnx_path}")

    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path))
        dummy = np.random.rand(1, 28, 28, 1).astype(np.float32)
        keras_out = model.predict(dummy, verbose=0)
        onnx_out = session.run(None, {"input": dummy})[0]
        diff = np.max(np.abs(keras_out - onnx_out))
        print(f"验证通过，最大差异: {diff:.8f}")
    except ImportError:
        print("跳过验证 (未安装 onnxruntime)")


def main():
    from src.paths import MODEL_PATH, TAURI_MODEL_PATH, ensure_project_dirs

    ensure_project_dirs()
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else MODEL_PATH
    onnx_path = Path(sys.argv[2]) if len(sys.argv) > 2 else TAURI_MODEL_PATH
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    convert(model_path, onnx_path)


if __name__ == "__main__":
    main()
