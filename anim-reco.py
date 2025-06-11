import tensorflow as tf
import numpy as np
from PIL import Image

# 加载预训练模型（基于ImageNet的397类动物分类）
model = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=True
)


def predict_animal(image_path):
    """输入动物图片，返回预测结果"""
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img)

    # 预处理（ImageNet标准）
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # 预测
    predictions = model.predict(img_array)
    decoded = tf.keras.applications.resnet50.decode_predictions(predictions, top=1)[0]
    return decoded[0][1]  # 返回动物名称（如'lion'）


# 示例
print(predict_animal("zebra.jpg"))  # 输出示例: 'zebra'