import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import cv2  # 用于图像预处理（可选）


class MNISTPredictor:
    def __init__(self):
        """自动下载MNIST数据和预训练模型"""
        # 加载数据
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

        # 构建并训练简单CNN模型（若需使用更复杂预训练模型可替换此部分）
        self.model = self._build_model()
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # 预处理数据
        x_train = self.x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
        y_train = to_categorical(self.y_train)

        # 训练模型（实际使用时建议保存模型避免重复训练）
        print("[MNIST] 正在训练轻量级模型（约1分钟）...")
        self.model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)
        print("[MNIST] 模型准备就绪！")

    def _build_model(self):
        """构建一个简单的CNN模型"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])
        return model

    def predict(self, image):
        """
        识别手写数字
        :param image: 28x28 numpy数组（0-255灰度值，黑底白字）
        :return: 预测的数字（0-9）
        """
        # 预处理
        img = image.reshape(1, 28, 28, 1).astype('float32') / 255
        # 预测
        return int(np.argmax(self.model.predict(img)))

    def test_random_sample(self):
        """随机测试一个样本"""
        idx = np.random.randint(0, len(self.x_test))
        digit = self.predict(self.x_test[idx])
        print(f"测试样本 {idx}: 预测={digit}, 实际={self.y_test[idx]}")
        return digit


# 全局实例（首次导入时会自动训练）
predictor = MNISTPredictor()

if __name__ == "__main__":
    # 示例：随机测试一个数字
    predictor.test_random_sample()