import cv2
import os
import numpy as np
from imgaug import augmenters as iaa


class ImageComparator:
    def __init__(self, training_dir):
        self.training_dir = training_dir
        self.training_features = {}  # 按类别存储直方图特征
        self.load_training_images()

    def load_training_images(self):
        """加载训练图像并提取直方图特征"""
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),  # 水平翻转
            iaa.Affine(rotate=(-20, 20))  # 旋转
        ])

        for category in os.listdir(self.training_dir):
            category_dir = os.path.join(self.training_dir, category)
            if os.path.isdir(category_dir):
                self.training_features[category] = []
                for image_name in os.listdir(category_dir):
                    image_path = os.path.join(category_dir, image_name)
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image = cv2.resize(image, (100, 100))

                    # 原始图像特征
                    hist = self.extract_features(image)
                    self.training_features[category].append(hist)

                    # 增强图像特征
                    augmented_image = seq.augment_image(image)
                    hist_aug = self.extract_features(augmented_image)
                    self.training_features[category].append(hist_aug)

    def extract_features(self, image):
        """提取灰度直方图特征（归一化）"""
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        return cv2.normalize(hist, hist).flatten()

    def predict(self, test_image_path):
        """通过直方图比对预测类别"""
        # 预处理测试图像
        test_image = cv2.imread(test_image_path)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        test_image = cv2.resize(test_image, (100, 100))
        test_hist = self.extract_features(test_image)

        # 比对所有训练样本
        min_distance = float('inf')
        best_category = "unknown"

        for category, features in self.training_features.items():
            distances = [
                cv2.compareHist(test_hist, train_hist, cv2.HISTCMP_CHISQR)
                for train_hist in features
            ]
            avg_distance = np.mean(distances)
            if avg_distance < min_distance:
                min_distance = avg_distance
                best_category = category

        return best_category

    def evaluate_accuracy(self):
        """评估方法需重写（此处省略实现）"""
        return 0.0  # 传统方法需自定义评估逻辑