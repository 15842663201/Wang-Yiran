import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
import random
from shutil import copyfile

# 设置数据路径
# data_dir = "F:\\NeRF\\InsightFace_NeRF\\insightface-master\\python-package\\insightface\\images\\vggface2_train\\vggface2_train\\train"
data_dir = "F:\\NeRF\\InsightFace_NeRF\\insightface-master\\python-package\\insightface\\images\\vggface2_test\\vggface2_test\\test"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# 初始化 FaceAnalysis
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# 定义训练集和测试集的划分比例
test_ratio = 0.2  # 每个人物文件夹中 20% 的图片用作测试

# 定义身份数据库
identity_database = {}

# 获取所有身份目录
person_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

for idx, person_id in enumerate(person_dirs):
    person_dir = os.path.join(data_dir, person_id)
    embeddings = []

    print(f"正在处理身份 {idx + 1}/{len(person_dirs)}: {person_id}")

    # 获取该文件夹内的所有图片路径
    img_files = os.listdir(person_dir)
    random.shuffle(img_files)  # 随机打乱图片

    # 划分为训练集和测试集
    split_index = int(len(img_files) * (1 - test_ratio))
    train_files = img_files[:split_index]  # 用于生成身份特征的图像
    test_files = img_files[split_index:]   # 用于后续识别准确率测试的图像

    # 处理训练集图片
    for img_idx, img_file in enumerate(train_files):
        img_path = os.path.join(person_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法加载图片 {img_path}")
            continue

        faces = app.get(img)
        if faces:
            embeddings.append(faces[0].embedding)
            print(f"  训练集图片 {img_idx + 1}/{len(train_files)} 已处理: {img_file}")

    # 计算平均特征向量，存入身份数据库
    if embeddings:
        identity_database[person_id] = np.mean(embeddings, axis=0)
        print(f"已完成身份: {person_id} 的特征向量生成")

    # 将测试集图片复制到 output/test_images 文件夹中以便后续测试
    for img_file in test_files:
        img_path = os.path.join(person_dir, img_file)
        test_img_dir = os.path.join(output_dir, "test_test_images", person_id)
        os.makedirs(test_img_dir, exist_ok=True)
        copyfile(img_path, os.path.join(test_img_dir, img_file))

# 保存训练集特征向量数据库为 .npy 文件
# np.save(os.path.join(output_dir, "custom_identity_embeddings_train.npy"), identity_database)
np.save(os.path.join(output_dir, "custom_identity_embeddings_test.npy"), identity_database)
print("训练集身份数据库已成功创建并保存！")
