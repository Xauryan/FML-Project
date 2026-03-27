import numpy as np
from keras.models import load_model

from src.paths import MODEL_PATH
from src.preprocessing import to_model_input


def load_digit_model(model_path=MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    return load_model(model_path, compile=False)


def predict_from_image(model, image):
    model_input = to_model_input(image)
    if np.count_nonzero(model_input) == 0:
        return None, 0.0, np.zeros(10, dtype=np.float32)

    probs = model.predict(model_input, verbose=0)[0]
    digit = int(np.argmax(probs))
    return digit, float(probs[digit]), probs
