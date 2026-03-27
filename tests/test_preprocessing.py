import numpy as np
from PIL import Image, ImageDraw

from src.preprocessing import preprocess_grayscale_array, preprocess_pil_image


def test_blank_image_returns_zeros():
    blank = np.full((64, 64), 255, dtype=np.uint8)
    processed = preprocess_grayscale_array(blank)

    assert processed.shape == (28, 28)
    assert np.count_nonzero(processed) == 0


def test_off_center_digit_is_recentred():
    canvas = Image.new("L", (80, 80), color=255)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((4, 20, 18, 56), fill=0)

    processed = preprocess_pil_image(canvas)
    ys, xs = np.indices(processed.shape)
    weight = processed.sum()
    center_x = float((processed * xs).sum() / weight)
    center_y = float((processed * ys).sum() / weight)

    assert processed.max() > 0.9
    assert 10.0 <= center_x <= 17.0
    assert 10.0 <= center_y <= 17.0
