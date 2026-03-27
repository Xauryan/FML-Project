
import tkinter as tk
from tkinter import filedialog
from keras.models import load_model
import numpy as np
from PIL import Image, ImageTk, ImageOps
import io
import os
os.environ['PATH'] += ';C:\\Program Files\\gs\\gs10.03.0\\bin\\'


# 加载保存的模型
model = load_model('lenet_mnist.h5')


def preprocess_image(image):
    image = image.convert('L').resize((28, 28))
    image = np.array(image)
    image = image.reshape(1, 28, 28, 1)
    image = image.astype('float32') / 255.0
    return image


def predict_digit_from_canvas():
    ps = canvas.postscript(colormode='gray')
    image = Image.open(io.BytesIO(ps.encode('utf-8')))
    # 将白底黑字转换为黑底白字
    image = ImageOps.invert(image)

    preprocessed_image = preprocess_image(image)
    result = model.predict(preprocessed_image)
    digit = np.argmax(result)
    prediction_label.config(text=f"Predicted Digit: {digit}")


def predict_digit_from_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path).convert('L').resize((28, 28))
        image_for_display = image.copy()
        photo = ImageTk.PhotoImage(image_for_display)
        image_label.config(image=photo)
        image_label.image = photo
        preprocessed_image = np.array(image).reshape(1, 28, 28, 1) / 255.0
        result = model.predict(preprocessed_image)
        digit = np.argmax(result)
        prediction_label.config(text=f"Predicted Digit: {digit}")


def clear_canvas():
    canvas.delete("all")
    prediction_label.config(text="Predicted Digit: None")
    image_label.config(image="")


def save_image():
    ps = canvas.postscript(colormode='gray')
    image = Image.open(io.BytesIO(ps.encode('utf-8')))
    image = image.point(lambda x: 255 if x < 128 else 0)
    image.save("drawn_digit.png")
    save_label.config(text="Image saved as 'drawn_digit.png'")


def draw(event):
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    canvas.create_oval(x1, y1, x2, y2, fill="black", width=10)


# 创建GUI界面
root = tk.Tk()
root.title("MNIST Digit Prediction")

# 创建清除按钮
clear_button = tk.Button(root, text="Clear", command=clear_canvas)
clear_button.pack(pady=20)

# 添加Canvas组件用于绘制数字
canvas = tk.Canvas(root, width=280, height=280, bg="white")
canvas.pack(pady=20)
canvas.bind("<B1-Motion>", draw)

# 添加预测按钮
predict_button = tk.Button(root, text="Predict Digit (Canvas)", command=predict_digit_from_canvas)
predict_button.pack(pady=20)

# 添加选择图片预测按钮
select_image_button = tk.Button(root, text="Select Image", command=predict_digit_from_file)
select_image_button.pack(pady=20)

# 添加保存按钮
save_button = tk.Button(root, text="Save Image", command=save_image)
save_button.pack(pady=20)

# 显示预测结果的标签
prediction_label = tk.Label(root, text="Predicted Digit: None", font=("Helvetica", 18))
prediction_label.pack(pady=20)

# 显示选择的图片的标签
image_label = tk.Label(root)
image_label.pack(pady=20)

# 显示保存信息的标签
save_label = tk.Label(root, text="", font=("Helvetica", 12))
save_label.pack(pady=20)

root.mainloop()