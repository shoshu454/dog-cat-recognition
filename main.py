from PySide2.QtUiTools import loadUiType, QUiLoader
from PySide2.QtCore import QFile, Qt, QTimer, QDateTime
from PySide2.QtGui import QIcon, QImage, QPixmap
from PySide2.QtWidgets import (QWidget, QTableWidgetItem, QApplication,
                               QDialog, QVBoxLayout, QLabel, QPushButton,
                               QGroupBox, QHBoxLayout, QFrame)
import cv2
import numpy as np
import os
from SVM import SVMBinaryClassifier

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog


# class SimplesvmClassifier:
#     def __init__(self):
#         self.X = []  # 特征向量
#         self.y = []  # 标签
#         self.model = KNeighborsClassifier(n_neighbors=3)  # 初始化svm模型，默认邻居数为3
#         self.scaler = StandardScaler()  # 用于数据标准化
#         self.trained = False
#
#     def extract_features(self, image_path):
#         """提取HOG特征"""
#         img = cv2.imread(image_path)
#         img = cv2.resize(img, (100, 100))
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         features = hog(gray,
#                        orientations=9,
#                        pixels_per_cell=(8, 8),
#                        cells_per_block=(2, 2),
#                        visualize=False)
#         return features
#
#     def add_sample(self, image_path, label):
#         """添加样本（特征 + 标签）"""
#         features = self.extract_features(image_path)
#         self.X.append(features)
#         self.y.append(label)
#         self.trained = False  # 新增样本后需重新训练
#
#     def train(self):
#         if len(self.X) < 2:
#             raise ValueError("至少需要两个样本才能训练模型")
#
#         X_scaled = self.scaler.fit_transform(self.X)
#         self.model.fit(X_scaled, self.y)
#         self.trained = True
#
#     def predict(self, image_path):
#         """预测样本类别"""
#         if not self.trained:
#             self.train()  # 若未训练则先训练模型
#
#         features = self.extract_features(image_path)
#         features_scaled = self.scaler.transform([features])
#         return self.model.predict(features_scaled)[0]
#
#     def evaluate_accuracy(self):
#         if len(self.X) < 2:
#             raise ValueError("至少需要两个样本才能评估")
#         if not self.trained:
#             self.train()
#
#             # 使用已拟合的scaler进行转换
#         X_scaled = self.scaler.transform(self.X)
#
#         # 使用交叉验证评估模型
#         scores = cross_val_score(self.model, X_scaled, self.y, cv=5)
#         return scores.mean()

class Gui(QWidget):
    def __init__(self):
        super().__init__()
        QtFileObj = QFile("recognize.ui")
        QtFileObj.open(QFile.ReadOnly)
        QtFileObj.close()
        self.ui = QUiLoader().load(QtFileObj)
        self.ui.pushButton.setText("capture")
        self.ui.pushButton.clicked.connect(self.capture_photo)

        self.training_dir = "training_data"  # 训练数据根目录
        self.clear_training_data()
        os.makedirs(self.training_dir, exist_ok=True)  # 创建文件夹
        self.photo_path = ""
        self.mode = "annotation"

        self.svm = SVMBinaryClassifier()
        self.ui.photoLabel = self.ui.findChild(QLabel, "photoLabel")
        self.ui.photoLabel.setText("拍摄的照片将显示在这里")
        self.ui.photoLabel.setAlignment(Qt.AlignCenter)
        self.ui.photoLabel.setMinimumSize(320, 240)
        self.ui.resultGroupBox = self.ui.findChild(QGroupBox, "resultGroupBox")
        self.ui.resultLayout = self.ui.findChild(QVBoxLayout, "resultLayout")
        self.ui.resultLabel = self.ui.findChild(QLabel, "resultLabel")
        self.ui.resultLabel.setText("等待操作...")
        self.ui.resultLabel.setAlignment(Qt.AlignCenter)
        self.ui.resultLabel.setStyleSheet("font-size: 16px; font-weight: bold;")
        if self.ui.resultLayout:
            self.ui.resultLayout.addWidget(self.ui.resultLabel)

        # # 变量定义、ui组件对象属性设置
        # self.index = 0
        # self.ui.tableWidgetAnswer.horizontalHeader().setVisible(True)  # 设置tableWidget组件的标题显示为True
        # self.ui.pushButton.clicked.connect(self.rename)  # 绑定按钮的方法
        self.cap = cv2.VideoCapture(0)  # 0 is the default camera
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_camera_view)
        self.timer.start(30)
        # self.ui.label.setText("jo")
        self.ui.modeButton = QPushButton("切换到识别模式")
        self.ui.modeButton.clicked.connect(self.toggle_mode)
        self.ui.verticalLayout.addWidget(self.ui.modeButton)
        # self.ui.photoLabel.setMinimumSize(320, 240)
        # self.ui.photoLabel.setFrameShape(QFrame.StyledPanel)  # 添加边框以便于调试
        # self.ui.photoLabel.setText("拍摄的照片将显示在这里")

    def clear_training_data(self):
        print("正在清理之前的训练数据...")
        try:
            if os.path.exists(self.training_dir):
                # 删除目录下的所有文件和子目录
                for item in os.listdir(self.training_dir):
                    item_path = os.path.join(self.training_dir, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        # 递归删除子目录
                        for sub_item in os.listdir(item_path):
                            sub_path = os.path.join(item_path, sub_item)
                            if os.path.isfile(sub_path):
                                os.remove(sub_path)
                        os.rmdir(item_path)
                print("训练数据清理完成")
            else:
                print("训练数据文件夹不存在，跳过清理")
        except Exception as e:
            print(f"清理训练数据时出错: {e}")

    def rename(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            # Convert the frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.ui.label.setPixmap(pixmap)

    def update_camera_view(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            # Convert the frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.ui.label.setPixmap(pixmap)

    def capture_photo(self):
        ret, frame = self.cap.read()
        if ret:
            # ========== 新增：生成唯一文件名 ==========
            timestamp = QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss.zzz")
            self.photo_path = f"{self.training_dir}/photo_{timestamp}.jpg"

            # 保存照片（不覆盖）
            cv2.imwrite(self.photo_path, frame)
            print(f"照片已保存：{self.photo_path}")

            # 显示照片
            pixmap = QPixmap(self.photo_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    self.ui.photoLabel.width(),
                    self.ui.photoLabel.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.ui.photoLabel.setPixmap(scaled_pixmap)
            else:
                print("错误：无法加载照片！")

            if self.mode == "annotation":
                self.show_annotation_dialog()
            else:
                self.perform_recognition()

    def toggle_mode(self):
        if self.mode == "recognition":
            self.mode = "annotation"
            self.ui.modeButton.setText("切换到识别模式")
            self.ui.resultLabel.setText("当前模式：标注")
        else:
            self.mode = "recognition"
            self.ui.modeButton.setText("切换到标注模式")
            self.ui.resultLabel.setText("当前模式：识别")
            try:
                accuracy = self.svm.evaluate_accuracy()
                print(f"模型当前准确度：{accuracy}")
            except ValueError as e:
                print(e)

    def show_annotation_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("标注照片")
        dialog.setMinimumWidth(300)

        layout = QVBoxLayout(dialog)

        # 显示照片预览
        photo_label = QLabel()
        if os.path.exists(self.photo_path):
            pixmap = QPixmap(self.photo_path)
            scaled_pixmap = pixmap.scaled(300, 200, Qt.KeepAspectRatio)
            photo_label.setPixmap(scaled_pixmap)
        else:
            photo_label.setText("照片加载失败")
        layout.addWidget(photo_label)

        # 添加标注选项
        label = QLabel("这是猫还是狗？")
        layout.addWidget(label)

        button_layout = QHBoxLayout()

        cat_button = QPushButton("🐱 猫")
        cat_button.setStyleSheet("padding: 10px; font-size: 14px;")
        cat_button.clicked.connect(lambda: self.save_annotation("cat", dialog))
        button_layout.addWidget(cat_button)

        dog_button = QPushButton("🐶 狗")
        dog_button.setStyleSheet("padding: 10px; font-size: 14px;")
        dog_button.clicked.connect(lambda: self.save_annotation("dog", dialog))
        button_layout.addWidget(dog_button)

        layout.addLayout(button_layout)

        dialog.exec_()

    def save_annotation(self, label, dialog):
        if not self.photo_path:
            self.ui.resultLabel.setText("错误：未捕获照片")
            dialog.close()
            return

            # ========== 新增：按类别保存照片 ==========
        category_dir = f"{self.training_dir}/{label}"
        os.makedirs(category_dir, exist_ok=True)
        new_path = f"{category_dir}/photo_{os.path.basename(self.photo_path)}"
        os.rename(self.photo_path, new_path)  # 移动到类别文件夹

        # 添加到训练集（使用新路径）
        self.svm.add_sample(new_path, label)
        self.ui.resultLabel.setText(f"已标注为：{label}，保存至：{new_path}")
        dialog.close()

    def perform_recognition(self):
        # 执行识别
        result = self.svm.predict(self.photo_path)

        # 更新结果显示
        result_text = f"识别结果：{result}"
        self.ui.resultLabel.setText(result_text)

    def closeEvent(self, event):
        # Release the camera when the window is closed
        self.cap.release()
        super().closeEvent(event)

    def logger_show(self):
        # 插入内容
        logger_item = {
            'one': '-' * 20, 'two': '-' * 20, 'three': '-' * 20, 'four': '-' * 20,
            'five': ''
        }
        self.ui.tableWidgetAnswer.insertRow(int(self.ui.tableWidgetAnswer.rowCount()))
        self.index += 1
        new_item_one = QTableWidgetItem(logger_item['one'])
        new_item_one.setTextAlignment(Qt.AlignCenter)
        new_item_two = QTableWidgetItem(logger_item['two'])
        new_item_two.setTextAlignment(Qt.AlignCenter)
        new_item_three = QTableWidgetItem(logger_item['three'])
        new_item_three.setTextAlignment(Qt.AlignCenter)
        new_item_four = QTableWidgetItem(logger_item['four'])
        new_item_four.setTextAlignment(Qt.AlignCenter)
        new_item_five = QTableWidgetItem(logger_item['five'])
        new_item_five.setTextAlignment(Qt.AlignCenter)
        self.ui.tableWidgetAnswer.setItem(self.index - 1, 0, new_item_one)
        self.ui.tableWidgetAnswer.setItem(self.index - 1, 1, new_item_two)
        self.ui.tableWidgetAnswer.setItem(self.index - 1, 2, new_item_three)
        self.ui.tableWidgetAnswer.setItem(self.index - 1, 3, new_item_four)
        self.ui.tableWidgetAnswer.setItem(self.index - 1, 4, new_item_five)
        # 定位至最新行
        self.ui.tableWidgetAnswer.verticalScrollBar().setSliderPosition(self.index)
        # 刷新
        QApplication.processEvents()


def recognize():
    print("Recognizing...")




# def main():
#     recognize()


if __name__ == "__main__":
    app = QApplication([])
    gui = Gui()
    gui.ui.show()
    app.exec_()