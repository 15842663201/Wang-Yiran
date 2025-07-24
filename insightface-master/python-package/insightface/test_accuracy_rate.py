import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
import os

# 设置路径
# train_embedding_path = "output/custom_identity_embeddings_train.npy"  # 训练集特征向量数据库路径
train_embedding_path = "output/custom_identity_embeddings_test.npy"  # 训练集特征向量数据库路径

test_img_dir = "output/test_test_images"  # 测试集图片路径

# 加载训练集特征向量数据库
train_embeddings = np.load(train_embedding_path, allow_pickle=True).item()

# 初始化 FaceAnalysis
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# 统计变量
correct_count = 0
total_count = 0
results = []

# 遍历测试集中的每个人物文件夹
for person_id in os.listdir(test_img_dir):
    person_dir = os.path.join(test_img_dir, person_id)
    if not os.path.isdir(person_dir):
        continue

    folder_correct = 0  # 统计每个文件夹的正确识别数
    folder_total = 0    # 每个文件夹的总图片数

    for img_file in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法加载图片 {img_path}")
            continue

        faces = app.get(img)
        if not faces:
            print(f"未检测到人脸: {img_path}")
            continue

        test_embedding = faces[0].embedding
        predicted_id = "Unknown"
        min_distance = float("inf")

        # 比对测试特征向量与训练特征向量
        for train_id, train_embedding in train_embeddings.items():
            distance = cosine(test_embedding, train_embedding)
            if distance < min_distance and distance < 0.6:  # 0.6 为相似度阈值
                min_distance = distance
                predicted_id = train_id

        # 判断识别结果是否正确
        if predicted_id == person_id:
            folder_correct += 1
            correct_count += 1

        folder_total += 1
        total_count += 1
        print(f"图片: {img_path}, 实际ID: {person_id}, 预测ID: {predicted_id}")

    # 计算并保存每个文件夹的准确率
    folder_accuracy = folder_correct / folder_total if folder_total > 0 else 0
    results.append(f"{person_id} 文件夹准确率: {folder_accuracy * 100:.2f}% ({folder_correct}/{folder_total})")

# 计算总体准确率
overall_accuracy = correct_count / total_count if total_count > 0 else 0
results.append(f"\n总体识别准确率: {overall_accuracy * 100:.2f}% ({correct_count}/{total_count})")

# 将结果保存到 txt 文件中
#with open("output/test_accuracy_results.txt", "w") as f:
#    f.write("\n".join(results))
with open("output/test_test_accuracy_results.txt", "w") as f:
    f.write("\n".join(results))

#print("\n测试完成，结果已保存到 output/test_accuracy_results.txt")
print("\n测试完成，结果已保存到 output/test_test_accuracy_results.txt")
