import tensorflow as tf
import numpy as np
from PIL import Image

# 加载预训练模型（植物病害分类，38类）
model = tf.keras.models.load_model('plant_disease_efficientnet.h5')


def predict_plant(image_path):
    """输入植物叶片图片，返回预测结果"""
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_names = ['Apple_healthy', 'Apple_scab', 'Tomato_early_blight', ...]  # 38类标签
    return class_names[np.argmax(predictions)]


# 示例：预测一张叶片图片
print(predict_plant("tomato_leaf.jpg"))  # 输出示例: "Tomato_early_blight"