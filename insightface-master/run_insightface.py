import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# 初始化 FaceAnalysis 并加载识别模型
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# 加载目标图像
img = cv2.imread('/Users/xuhaoyang/Desktop/NeRF/insightface-master/python-package/insightface/data/images/t2.jpg')  # 这里可以替换成你本地的图片路径
faces = app.get(img)

# 遍历检测到的所有人脸，提取人脸特征
for face in faces:
    print("Face ID embedding:", face.embedding) #这是人脸特征向量
    # 可以在此处将 embedding 与数据库中的人脸特征进行比对来确认身份
    embedding = face.embedding  # 提取特征向量
    # 比对 embedding 和数据库中的向量，找到最接近的身份
    # 将识别出的身份标注在图片上

# 将识别结果绘制在图像上并保存
rimg = app.draw_on(img, faces)
cv2.imwrite("./t2_output_with_id.jpg", rimg)
