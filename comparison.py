import cv2
from skimage.metrics import structural_similarity as ssim
import os
import numpy as np
from sklearn.model_selection import train_test_split


class ImageComparator:
    def __init__(self, training_dir):
        self.training_dir = training_dir
        self.training_images = {}
        self.load_training_images()

    def evaluate_accuracy(self):
        """评估准确率"""
        X = []
        y = []
        for category, images in self.training_images.items():
            for image in images:
                X.append(image)
                y.append(category)

        if len(np.unique(y)) < 2:
            raise ValueError("至少需要两个类别的样本才能评估")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        correct_predictions = 0
        for i in range(len(X_test)):
            test_image = X_test[i]
            true_label = y_test[i]

            max_similarity = -1
            predicted_category = None

            for j in range(len(X_train)):
                training_image = X_train[j]
                training_label = y_train[j]
                similarity = self.compare_images(test_image, training_image)
                if similarity > max_similarity:
                    max_similarity = similarity
                    predicted_category = training_label

            if predicted_category == true_label:
                correct_predictions += 1

        accuracy = correct_predictions / len(X_test)
        return accuracy

    def load_training_images(self):
        """加载训练集中的图像"""
        for category in os.listdir(self.training_dir):
            category_dir = os.path.join(self.training_dir, category)
            if os.path.isdir(category_dir):
                self.training_images[category] = []
                for image_name in os.listdir(category_dir):
                    image_path = os.path.join(category_dir, image_name)
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image = cv2.resize(image, (100, 100))
                    self.training_images[category].append(image)

    def compare_images(self, image1, image2):
        """计算两张图像的 SSIM 值"""
        return ssim(image1, image2)

    def predict(self, test_image_path):
        """预测测试图像的类别"""
        test_image = cv2.imread(test_image_path)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        test_image = cv2.resize(test_image, (100, 100))

        max_similarity = -1
        predicted_category = None

        for category, images in self.training_images.items():
            for training_image in images:
                similarity = self.compare_images(test_image, training_image)
                if similarity > max_similarity:
                    max_similarity = similarity
                    predicted_category = category

        return predicted_category