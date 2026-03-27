import numpy as np
from PIL import Image

OUTPUT_SIZE = 28
CONTENT_SIZE = 20
INK_THRESHOLD = 250


def _resize_filter():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS


def preprocess_grayscale_array(
    image_array,
    *,
    output_size=OUTPUT_SIZE,
    content_size=CONTENT_SIZE,
    ink_threshold=INK_THRESHOLD,
):
    """Convert a white-background digit image into an MNIST-like tensor."""
    image_array = np.asarray(image_array, dtype=np.uint8)
    if image_array.ndim != 2:
        raise ValueError("expected a 2D grayscale image array")

    mask = image_array < ink_threshold
    if not np.any(mask):
        return np.zeros((output_size, output_size), dtype=np.float32)

    ys, xs = np.where(mask)
    cropped = image_array[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    cropped_image = Image.fromarray(cropped, mode="L")

    scale = content_size / max(cropped.shape)
    resized_width = max(1, int(round(cropped.shape[1] * scale)))
    resized_height = max(1, int(round(cropped.shape[0] * scale)))
    resized = cropped_image.resize((resized_width, resized_height), _resize_filter())

    canvas = Image.new("L", (output_size, output_size), color=255)
    offset_x = (output_size - resized_width) // 2
    offset_y = (output_size - resized_height) // 2
    canvas.paste(resized, (offset_x, offset_y))

    normalized = (255.0 - np.asarray(canvas, dtype=np.float32)) / 255.0
    return normalized


def preprocess_pil_image(
    image,
    *,
    output_size=OUTPUT_SIZE,
    content_size=CONTENT_SIZE,
    ink_threshold=INK_THRESHOLD,
):
    grayscale = image.convert("L")
    return preprocess_grayscale_array(
        grayscale,
        output_size=output_size,
        content_size=content_size,
        ink_threshold=ink_threshold,
    )


def to_model_input(image):
    normalized = preprocess_pil_image(image)
    return normalized.reshape(1, OUTPUT_SIZE, OUTPUT_SIZE, 1)
