# 这是InsightFace用于我们NeRF的相关方法
~~需要注意的是，目前本项目只在macos上进行了部署，windows可能会有所不同。~~
# Updating log
- 2024.11.15
  - HYXU6
    - 成功在Windows系统上部署本项目，同时为了使用GPU来读取、识别和处理图像，以后的相关任务均迁移到Windows上进行。
    - 更新create_identity_database.py和test_accuracy_rate.py，用于生成npy文件和测试insightface准确性。
    - 完成基于vggface2_train和vggface2_test的测试。得出结果是train准确率95.74%，test准确率93.78%。
    - 测试方法：介于vggface2_train和vggface2_test的目录结构一致，均为内部有若干带有唯一编号的文件夹，其中有数百张同一人脸的jpg格式的图片。然而train和test中没有相同人脸出现，故将每个人的照片随机抽取80%用于生成特征向量，剩余20%测试是否能够正确识别。
- 2024.11.4
  - HYXU6
    - 完成vggface2_train中的图片读取工作。
    - 该工作仍然使用macos环境，使用cpu，连续读取时间34h。
    - 成功创建custom_identity_embeddings.npy文件，用来识别人脸及其id。需要注意的是，每个记录的人脸的id就是vggface2_train中相应地址的id。
  - VIb010
    - 提交并上传idr相关代码。
- 2024.11.2
  - HYXU6
    - 创建该项目并实现insightface的复现

# Ready for start
## 1.确保安装必要python库
```bash
pip install insightface numpy scipy opencv-python
```
## 2.摘取自insightface，需要手动安装onnxruntime。
建议使用搭载英伟达显卡的设备。
Install Inference Backend
For insightface<=0.1.5, we use MXNet as inference backend.
Starting from insightface>=0.2, we use onnxruntime as inference backend.
You have to install onnxruntime-gpu manually to enable GPU inference, or install onnxruntime to use CPU only inference.


# Quick Start
- 在/NeRF/insightface-master中启用终端执行以下命令
```bash
python3 run_insightface.py
```
- 这将会输出一个名称为t2_output_with_id.jpg的图像。其中人脸被标识为相应的预测性别和年龄
  - M = Male
  - F = Female
  - 后续数字为预测年龄
- 如果想要t1_output.jpg结果，请移步至原项目的quick start
 

# Start for id_detecting
- 在/NeRF/insightface-master/python-package/insightface中启用终端执行以下命令
```bash
python3 create_identity_database.py
python3 recognize_identity.py
```
- create_identity_database.py的功能：
  - 初始化InsightFace
  - 定义已知身份及其照片路径
  - 提取特征向量并存储到数据库
  - 保存特征向量数据库到`.npy`文件
    - 该文件保存在output中
- recognize_identity.py的功能：
  - 初始化 FaceAnalysis
  - 加载身份数据库
  - 加载要识别的目标图像
  - 识别并比对身份
  - 保存识别后的图像
    - 该图像保存在output中
   
# 关于调整和优化
1. 相似度阈值调整：distance < 0.6 是一个默认值，根据实际识别效果可以适当调高或调低。
2. 丰富数据库：每个人可使用多张不同角度、表情的照片，以提高识别准确率。
3. 优化存储格式：如果数据库较大，建议使用 SQLite 或其他数据库系统来管理特征向量和身份信息。
