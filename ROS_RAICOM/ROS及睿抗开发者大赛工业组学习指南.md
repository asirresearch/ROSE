# 睿抗工业组学习指南

## 1. ROS 快速入门教程
这是一个对新手友好、易于吸收的视频教程。在开始进行比赛核心内容开发之前，强烈建议完整观看一遍，建立对 ROS 节点、话题、服务的基本概念：
*   **【机器人操作系统 ROS 快速入门教程】**：[https://www.bilibili.com/video/BV1BP4y1o7pw/](https://www.bilibili.com/video/BV1BP4y1o7pw/)

## 2. 元创兴官方提供的工业组学习路径
官方教程提供了打包好的虚拟机镜像、ROS 环境以及比赛用到的所有机器人底层控制与导航功能包。
课程链接获取：[https://reinovo.yuque.com/dcddrf/sa3u5y/wm7p846c8xvum4ei?singleDoc#]
请严格按照此文档配置虚拟机：[https://reinovo.yuque.com/dcddrf/sa3u5y/bg3gvpbawcex95kn]
---

## 3. 识别与抓取

### 3.1 YOLOv8环境配置

#### 1. 安装pytorch

*  说明：根据你的环境（CPU / CUDA），选择合适的 PyTorch 安装命令。若使用 GPU，请参考 https://pytorch.org 上匹配 CUDA 的安装指导。

*  在 CPU-only 环境安装 PyTorch：
如果有 CUDA（GPU），请在 https://pytorch.org 选择匹配的安装命令（示例需根据你的 CUDA 版本调整）。
*  在 CPU-only 环境安装 PyTorch：
!pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

#### 2. 安装YOLOv8

说明：Ultralytics 提供的 `ultralytics` 包含 YOLOv8 的训练、推理与导出功能。安装后可使用 `yolo` CLI 或 Python API。

安装 ultralytics（包含 yolo CLI 与 Python API）：
!pip3 install ultralytics

#### 3. 验证安装

说明：下面的示例命令会用示例图片运行一次推理，验证 `yolo` CLI 与模型可用性。

使用 yolo CLI 在示例图片上跑一次推理（若本地没有模型，CLI 会自动下载 yolov8n.pt）：
运行后将在 runs/predict 下生成预测结果，可查看图片或保存的推理可视化：
!yolo predict model=yolov8n.pt source="https://ultralytics.com/images/bus.jpg"
打印下载版本：
import torch
print('torch version:', torch.__version__)
print('cuda available:', torch.cuda.is_available(), 'device count:', torch.cuda.device_count())
try:
    import ultralytics
    print('ultralytics version:', ultralytics.__version__)
except Exception as e:
    print('ultralytics import error:', e)

### 3.2 数据标注

#### 1. 下载标注软件 labelme

- 推荐安装方式：

```bash
# 使用 pip
pip install labelme==5.5.0
```
- 启动：`labelme`，在界面中绘制多边形（用于分割）或矩形（用于检测/定位），保存为 `.json` 文件。

![](images/2026-05-28-10-31-03.png)

- 打开数据图像目录

![](images/2026-05-28-10-33-18.png)

- 设置json数据输出路径，关闭同时保存图像数据

![](images/2026-05-28-10-34-16.png)

- 开启自动保存数据

![](images/2026-05-28-10-36-50.png)

#### 2. 任务分析

- 明确任务类型：分割（segmentation）或分类（classification）。
- 确定类别（labels）列表与命名规范，写入 `labels.txt` 或项目文档。
- 确定数据量和标注粒度（是否需要 instance-level、边界处理、遮挡标注规则）。
- 质检规则：复核比例、仲裁人员与标注一致性检查方法。

#### 3. 分割任务数据标注

##### 1. 创建数据库

- 目录结构示例：

```
./datasets                    # 数据集根目录（所有训练数据放在这里）
└── Industrial_sign           # 工业标识 数据集（你的分割项目）
    ├── classes.txt           # 类别名称列表（一行一个类别，用于转换脚本）
    ├── images                # 原始图片文件夹（.jpg/.png）
    ├── json_seg              # LabelMe 标注的 JSON 分割文件（多边形标注）
    ├── labels                # 转换后的 YOLO 分割标签（.txt 格式）
    ├── polygon2YOLOseg.py    # 脚本：JSON 多边形 → YOLO 分割格式 txt
    ├── spilt_data.py         # 脚本：随机划分训练集 train / 验证集 val
    ├── train.txt             # 训练集图片路径列表
    └── val.txt               # 验证集图片路径列表
└── Industrial_sign.yaml      # YOLO 训练配置文件（核心！训练时加载）
```

- 每张图片用 `labelme` 生成 `.json`，保存多边形标注（字段 `shapes`）。

![](images/2026-05-28-10-48-16.png)

##### 2 数据转换

- 常见目标：将 `labelme` 的 `.json` 转为训练框架可读格式（COCO / VOC / YOLO segmentation）。
- 使用 `labelme` 自带或社区脚本转换为 COCO：

```python
import os
import json
import argparse

def labelme2yoloseg(json_dir, save_dir, class_txt):
    """
    LabelMe JSON 批量转 YOLO-seg 格式
    :param json_dir: JSON 文件所在文件夹
    :param save_dir: 输出 TXT 标签保存文件夹
    :param class_txt: 类别文件（每行一个类别，从0开始编号）
    """
    # 读取类别列表
    with open(class_txt, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    
    # 创建输出目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 遍历所有 JSON 文件
    for json_name in os.listdir(json_dir):
        if not json_name.endswith('.json'):
            continue
        
        json_path = os.path.join(json_dir, json_name)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 获取图片宽高（用于归一化）
        img_h = data['imageHeight']
        img_w = data['imageWidth']
        
        # 输出标签文件名（与图片同名）
        txt_name = os.path.splitext(json_name)[0] + '.txt'
        txt_path = os.path.join(save_dir, txt_name)
        
        # 逐行写入标签
        with open(txt_path, 'w', encoding='utf-8') as f:
            for shape in data['shapes']:
                label = shape['label']
                # 跳过不在类别列表里的标签
                if label not in classes:
                    continue
                
                # 获取类别 ID
                cls_id = classes.index(label)
                
                # 获取多边形点
                points = shape['points']
                
                # 归一化坐标并拼接
                seg_points = []
                for (x, y) in points:
                    x_norm = x / img_w
                    y_norm = y / img_h
                    seg_points.append(f"{x_norm:.6f}")
                    seg_points.append(f"{y_norm:.6f}")
                
                # 拼接成 YOLO-seg 行
                line = f"{cls_id} {' '.join(seg_points)}\n"
                f.write(line)
    
    print(f"✅ 转换完成！共处理 {len(os.listdir(json_dir))} 个文件")
    print(f"✅ 标签保存到：{save_dir}")
    print(f"✅ 类别列表：{classes}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_dir', default='json', help='LabelMe JSON 文件夹')
    parser.add_argument('--save_dir', default='labels', help='YOLO-seg 标签保存文件夹')
    parser.add_argument('--class_txt', default='classes.txt', help='类别文件路径')
    opt = parser.parse_args()
    
    labelme2yoloseg(opt.json_dir, opt.save_dir, opt.class_txt)
```

- 对于 YOLOv8 segmentation，通常需要生成 `data.yaml` 并将 masks 转为适配的格式（或使用 COCO 格式作为输入）。

示例 `data.yaml`（COCO 格式）:

```yaml
path: ./datasets/Industrial_sign
train: train.txt
val: val.txt

nc: 6
names:
  0: coms
  1: lens
  2: camera
  3: servo
  4: Chip
  5: Robor
```

##### 3 数据分集

- 常见划分：`train:val:test = 8:1:1` 或 `0.8/0.2`（含验证集）。
- 可用简单脚本按文件名随机划分：

```python
import os
import random

# 打开文件
train_file = open("train.txt", "w", encoding="utf-8")
val_file = open("val.txt", "w", encoding="utf-8")

def split_data(data, rate):
    random.shuffle(data)
    split_index = int(len(data) * rate)
    for i, img in enumerate(data):
        if i < split_index:
            train_file.write(img + '\n')
        else:
            val_file.write(img + '\n')

label_dir = "labels"       # 标签文件夹
image_dir = "./images"        # 图片文件夹
train_rate = 0.8                  # 训练集比例

img_list = []

# 遍历所有标签，自动匹配真实存在的图片
for txt_name in os.listdir(label_dir):
    if not txt_name.endswith(".txt"):
        continue
    
    # 获取文件名（不带后缀）
    base_name = os.path.splitext(txt_name)[0]
    
    # 遍历所有可能的图片后缀
    for suffix in [".JPG", ".jpg", ".jpeg", ".png", ".JPEG"]:
        img_name = base_name + suffix
        img_path = os.path.join(image_dir, img_name)
        
        # 如果找到真实存在的图片，就加入列表
        if os.path.exists(img_path):
            img_list.append(img_path)
            break

# 划分数据并生成 txt
split_data(img_list, train_rate)

# 关闭文件
train_file.close()
val_file.close()

print(f"✅ 处理完成！共找到 {len(img_list)} 张图片")
print("✅ train.txt / val.txt 路径 100% 正确！")
```

- 确保对应的 annotation `.json` / mask 文件一并移动或转换后同步路径。

##### 4 模型训练

- 使用 YOLOv8 segmentation 示例命令（基于 ultralytics yolo CLI）：

```bash
yolo train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640 batch=2
```
- - train：训练模式
- -  model=yolov8n.pt：使用 YOLOv8n 最小模型（速度快、占显存小）
- -  data=...yaml：指定你的数据集配置
- -  epochs=100：训练 100 轮
- - imgsz=640：输入图像尺寸 640
- - batch=2：批次 2（适合小显存显卡，不会爆显存）

### 3.3 工业物块抓取

#### 1.开启前置节点
开启导航
```bash
roslaunch oryxbot_navigation demo_nav_2d.launch
```
开启底盘相机识别
```bash
roslaunch ar_pose ar_base_sim.launch
```
开启相对移动
```bash
roslaunch relative_move relative_move.launch 
```
开启机械臂仿真
```bash
roslaunch oryxbot_description swiftpro_control.launch
```
开启手臂相机ar码识别
```bash
roslaunch ar_pose ar_hand_sim.launch
```
#### 2. REI Robot & Arm 仿真api文档
- REIRobotSim：底盘导航、相对移动、AR 二次定位
- REIArmSim：机械臂运动、气泵控制
- DynamicTFPublisher：动态 TF 坐标发布（视觉抓取专用）
import rospy
from reirobot_API import REIRobot_sim,REIArm_sim

rospy.init_node("oryxbot_sim_node",anonymous=True)
base = REIRobot_sim()
arm = REIArm_sim()

base.get_tf('map','base_footprint')
* 机器人导航到目标位置：
base.nav_to_goal([1.936,1.203,0,1])
对准工作台：
base.set_ARtrack(0,0.4)
base.set_relmove(0.18,0,0)
* 机械臂移动到观测位置：
arm.arm_move([200,-150,120])

#### 3.YOLO目标分割

* 获取图像数据并加载分割模型：
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from reirobot_API import DynamicTFPublisher
from ultralytics import YOLO

model = YOLO("./weights/Industrial.pt")
rei_tf = DynamicTFPublisher()
rei_tf.clear_all_tf()

import matplotlib.pyplot as plt

bridge = CvBridge()
image_msg = rospy.wait_for_message("/hand_camera/image_raw",Image)
cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
plt.imshow(cv_image[:,:,::-1])
plt.show()

* 识别图像：
results = model(cv_image, show=False, save=True)

* 获取分割结果：
names = [result.names for result in results]
print(names)
mask = results[0].masks[0].data.cpu().numpy().squeeze()
plt.imshow(mask,cmap="gray")
plt.show()

#### 4.获取角点

* 轮廓提取：
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = max(contours, key=cv2.contourArea)

img_contour = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
cv2.drawContours(img_contour, [cnt], -1, (0,255,0), 2)

plt.imshow(img_contour)
plt.axis("off")
* 多边形逼近获取轮廓角点：
epsilon = 0.02 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)

#坐标缩放回原图
h0, w0 = results[0].orig_shape                  # 原图高、宽
h_in, w_in = results[0].masks.shape[1:]         # mask 尺寸 (h, w)
scale_x = w0 / w_in                          # x 缩放比例
scale_y = h0 / h_in                          # y 缩放比例

#把 mask 上的角点 → 映射到原图坐标
approx_original = (approx * [scale_x, scale_y]).astype(np.int32)

img_approx = np.copy(cv_image)
cv2.drawContours(img_approx, [approx_original], -1, (0, 255, 0), 3)  # 绿色轮廓

for p in approx_original:
    cx, cy = p[0]
    cv2.circle(img_approx, (cx, cy), 8, (255, 0, 0), -1)  # 蓝色角点

plt.imshow(img_approx[..., ::-1])
plt.axis("off")

#### 5.pnp求解

PnP（Perspective-n-Point）：透视 n 点问题，已知世界坐标系下 n 个 3D 点坐标、图像坐标系下对应 2D 像素坐标、相机内参，求解相机外参（旋转矩阵 R + 平移向量 t），即相机在世界坐标系下的位姿。
#整理 2D 角点
image_points = np.array([
    approx_original[0][0], approx_original[1][0], approx_original[2][0], approx_original[3][0]
], dtype=np.float32)

#定义 3D 角点
square_size = 0.03
object_points = np.array([
    [-square_size/2,  square_size/2, 0],
    [ square_size/2,  square_size/2, 0],
    [ square_size/2, -square_size/2, 0],
    [-square_size/2, -square_size/2, 0],
], dtype=np.float64)

* 获取相机内参：
from sensor_msgs.msg import CameraInfo 
#相机内参
camera_info = rospy.wait_for_message('hand_camera/camera_info',CameraInfo)
camera_matrix = np.array(camera_info.K,np.float64).reshape(3,3)
dist_coeffs = np.array(camera_info.D, np.float64)
* pnp求解获取平移和旋转矩阵：
ret, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
R, _ = cv2.Rodrigues(rvec)
print("旋转矩阵 R:\n", np.round(R, 3))
print("平移 T (m):", np.round(tvec.flatten(), 3))
* 将旋转矩阵转为四元素：
from tf.transformations import quaternion_from_matrix
T = np.eye(4)
T[:3,:3] = R
q = quaternion_from_matrix(T)
print("四元素为：",q)
* 发布目标tf：
![image.png](attachment:image.png)
rei_tf.clear_all_tf()
rei_tf.add_dynamic_tf('hand_cam_link','target_1',tvec.ravel(),q)
* 移动到目标位置：
arm.arm_move([200,-150,120])
rospy.sleep(1)
pos,rot = base.get_tf("Base","target_1")
pos = [p*1000 for p in pos]
arm.arm_move(pos)