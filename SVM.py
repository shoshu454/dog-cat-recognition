# SVM.py
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib


class SVMBinaryClassifier:
    def __init__(self):
        self.X = []  # 特征向量
        self.y = []  # 标签
        self.scaler = StandardScaler()  # 标准化器
        self.model = SVC(kernel='rbf', C=1.0, gamma='scale')  # SVM模型
        self.trained = False  # 训练状态标志

    def extract_features(self, image_path):
        """提取HOG特征（与KNN保持完全一致）"""
        img = cv2.imread(image_path)
        img = cv2.resize(img, (128, 128))  # 确保尺寸与KNN一致
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = hog(
            gray,
            orientations=12,
            pixels_per_cell=(6, 6),
            cells_per_block=(3, 3),
            visualize=False,
            block_norm = 'L2-Hys'
        )
        return features

    def add_sample(self, image_path, label):
        """添加样本"""
        features = self.extract_features(image_path)
        self.X.append(features)
        self.y.append(label)
        self.trained = False  # 新增样本后需重新训练

    def train(self):
        """训练模型"""
        if len(np.unique(self.y)) < 2:
            raise ValueError("至少需要两个类别的样本才能训练")

        # 标准化处理
        X_scaled = self.scaler.fit_transform(self.X)
        self.model.fit(X_scaled, self.y)
        self.trained = True

    def predict(self, image_path):
        """预测类别"""
        if not self.trained:
            self.train()

        features = self.extract_features(image_path)
        features_scaled = self.scaler.transform([features])
        return self.model.predict(features_scaled)[0]

    def evaluate_accuracy(self):
        """评估准确率"""
        if not self.trained:
            self.train()

        X_scaled = self.scaler.transform(self.X)
        scores = cross_val_score(self.model, X_scaled, self.y, cv=3)
        return np.mean(scores)

    def save_model(self, path='svm_model.pkl'):
        """保存模型"""
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)

    def load_model(self, path='svm_model.pkl'):
        """加载模型"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.trained = True