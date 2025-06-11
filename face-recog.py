import cv2

# 加载预训练人脸检测模型（OpenCV自带）
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_faces(image_path, output_path='output.jpg'):
    # 读取图片
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,  # 检测尺度变化（值越小越敏感）
        minNeighbors=5,  # 检测框合并阈值
        minSize=(30, 30)  # 最小人脸尺寸
    )

    # 画方框标注
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 蓝色方框，线宽2px

    # 保存结果
    cv2.imwrite(output_path, img)
    print(f"检测到 {len(faces)} 张人脸，结果已保存至 {output_path}")
    return faces


# 使用示例
detect_faces("input.jpg")  # 替换为你的图片路径