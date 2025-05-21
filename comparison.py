import cv2
import os
import numpy as np
from imgaug import augmenters as iaa
from skimage.feature import hog


class ImageComparator:
    def __init__(self, training_dir):
        self.training_dir = training_dir
        self.training_features = {}  # 按类别存储直方图特征
        self.load_training_images()

    def load_training_images(self):
        """加载训练图像并提取直方图特征"""
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),  # 水平翻转
            iaa.Affine(rotate=(-20, 20)),  # 旋转
            iaa.Multiply((0.8, 1.2)),  # 亮度调整
            iaa.ContrastNormalization((0.8, 1.2))  # 对比度调整
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
    #     """提取HOG特征"""
        features = hog(image, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=False)
        return features

    def predict(self, test_image_path):
        """通过直方图比对预测类别"""
        # 预处理测试图像
        test_image = cv2.imread(test_image_path)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        test_image = cv2.resize(test_image, (100, 100))
        test_features = self.extract_features(test_image)

        # 比对所有训练样本
        min_distance = float('inf')
        best_category = "unknown"

        for category, features_list in self.training_features.items():
            distances = [
                np.linalg.norm(test_features - train_features)
                for train_features in features_list
            ]
            avg_distance = np.mean(distances)
            if avg_distance < min_distance:
                min_distance = avg_distance
                best_category = category

        return best_category

    def evaluate_accuracy(self):
        """评估方法需重写（此处省略实现）"""
        correct = 0
        total = 0

        for category in os.listdir(self.training_dir):
            category_dir = os.path.join(self.training_dir, category)
            if os.path.isdir(category_dir):
                for image_name in os.listdir(category_dir):
                    image_path = os.path.join(category_dir, image_name)
                    prediction = self.predict(image_path)
                    if prediction == category:
                        correct += 1
                    total += 1

        if total == 0:
            return 0.0
        return correct / total