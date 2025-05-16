import os
import shutil
import random
from pathlib import Path

# 配置参数
RAW_DATA_DIR = r'C:\Users\wucai\Desktop\神经网络\任务1\flower_dataset'  # 原始数据集根目录
OUTPUT_DIR = r'C:\Users\wucai\Desktop\神经网络\任务1\flower_imagenet' # 输出目录（ImageNet格式）
TRAIN_RATIO = 0.8  # 训练集比例
SEED = 42  # 随机种子（确保可复现性）

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
train_dir = os.path.join(OUTPUT_DIR, "train")
val_dir = os.path.join(OUTPUT_DIR, "val")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 获取所有类别名称
classes = [d.name for d in Path(RAW_DATA_DIR).iterdir() if d.is_dir()]
classes.sort()  # 确保顺序一致

# 生成类别文件 classes.txt
with open(os.path.join(OUTPUT_DIR, "classes.txt"), "w") as f:
    for cls in classes:
        f.write(f"{cls}\n")

# 拆分每个类别并复制文件
for cls in classes:
    cls_path = os.path.join(RAW_DATA_DIR, cls)
    images = [f for f in os.listdir(cls_path) if f.endswith((".jpg", ".jpeg", ".png"))]
    random.seed(SEED)
    random.shuffle(images)

    # 计算拆分索引
    split_idx = int(len(images) * TRAIN_RATIO)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # 创建类别子文件夹（train/val下）
    train_cls_dir = os.path.join(train_dir, cls)
    val_cls_dir = os.path.join(val_dir, cls)
    os.makedirs(train_cls_dir, exist_ok=True)
    os.makedirs(val_cls_dir, exist_ok=True)

    # 复制训练集文件
    for img in train_images:
        src = os.path.join(cls_path, img)
        dst = os.path.join(train_cls_dir, img)
        shutil.copyfile(src, dst)

    # 复制验证集文件
    for img in val_images:
        src = os.path.join(cls_path, img)
        dst = os.path.join(val_cls_dir, img)
        shutil.copyfile(src, dst)


# 生成标注文件（train.txt和val.txt）
def generate_annotation_file(data_dir, output_file):
    with open(output_file, "w") as f:
        for idx, cls in enumerate(classes):
            cls_dir = os.path.join(data_dir, cls)
            for img in os.listdir(cls_dir):
                f.write(f"{cls}/{img} {idx}\n")


# 生成验证集标注（必填）
val_ann_file = os.path.join(OUTPUT_DIR, "val.txt")
generate_annotation_file(val_dir, val_ann_file)

# 可选：生成训练集标注
train_ann_file = os.path.join(OUTPUT_DIR, "train.txt")
generate_annotation_file(train_dir, train_ann_file)