import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
import os

# 初始化 FaceAnalysis
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# 加载身份数据库
identity_database = np.load("output/custom_identity_embeddings.npy", allow_pickle=True).item()

# 加载要识别的目标图像（相对路径）
img = cv2.imread("images/0009_01.jpg")
if img is None:
    print("Error: Could not load image. Check the file path.")
    exit()

faces = app.get(img)

# 识别并比对身份
for face in faces:
    min_distance = float("inf")
    identity = "Unknown"
    for name, db_embedding in identity_database.items():
        distance = cosine(face.embedding, db_embedding)  # 计算余弦相似度
        if distance < min_distance and distance < 0.6:  # 0.6 为可调整阈值
            min_distance = distance
            identity = name

    # 将识别出的身份信息显示在图像上
    cv2.putText(img, f"Name: {identity}", (int(face.bbox[0]), int(face.bbox[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# 保存识别后的图像（相对路径）
os.makedirs("output", exist_ok=True)
cv2.imwrite("output/identified_image.jpg", img)

print("身份识别完成，结果已保存！")
