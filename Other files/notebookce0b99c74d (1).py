#!/usr/bin/env python
# coding: utf-8

# In[1]:


DATA_PATH = "../input/coco-mini-5000"
TRAIN_IMAGES = f"{DATA_PATH}/images/train2017"
VAL_IMAGES = f"{DATA_PATH}/images/val2017"
TRAIN_LABELS = f"{DATA_PATH}/labels/train2017"
VAL_LABELS = f"{DATA_PATH}/labels/val2017"


# In[2]:


get_ipython().system('pip install -q ultralytics')
from ultralytics import YOLO


# In[9]:


yaml_content = """
path: /kaggle/input/coco-mini-4000/coco-mini-4000
train: images/train2017
val: images/val2017
nc: 80
names: [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
         'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
         'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
         'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
         'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
         'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
         'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
         'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
         'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
         'toothbrush' ]
"""

with open("coco-mini-4000.yaml", "w") as f:
    f.write(yaml_content)


# In[8]:


get_ipython().system('ls /kaggle/input/coco-mini-5000/images')
get_ipython().system('ls /kaggle/input/coco-mini-5000/labels')


# In[11]:


from ultralytics import YOLO

model = YOLO("yolov8s.pt")

results = model.train(
    data="coco-mini-4000.yaml",
    epochs=15,
    imgsz=640,
    batch=8,
    project="kaggle_yolo_training",
    name="coco_mini_4000_run",
    save=True
)


# In[17]:


import matplotlib.pyplot as plt

plt.figure(figsize=(16, 8))

# Plot Training Losses
plt.subplot(1, 2, 1)
plt.plot(df['epoch'], df['train/box_loss'], label='Box Loss')
plt.plot(df['epoch'], df['train/cls_loss'], label='Class Loss')
plt.plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss')
plt.title("Training Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Plot Evaluation Metrics
plt.subplot(1, 2, 2)
plt.plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
plt.plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
plt.title("Evaluation Metrics")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()

plt.tight_layout()
plt.show()


# In[18]:


get_ipython().system('pip install -q ultralytics')


# In[20]:


import os
import glob
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import classification_report

# Load model
model = YOLO("/kaggle/working/kaggle_yolo_training/coco_mini_4000_run2/weights/best.pt")

# Dataset paths
val_img_dir = "/kaggle/input/coco-mini-4000/coco-mini-4000/images/val2017"
val_lbl_dir = "/kaggle/input/coco-mini-4000/coco-mini-4000/labels/val2017"

y_true = []
y_pred = []

# Collect validation images
image_paths = sorted(glob.glob(os.path.join(val_img_dir, "*.jpg")))

for img_path in image_paths:
    base_name = os.path.basename(img_path).replace(".jpg", ".txt")
    label_path = os.path.join(val_lbl_dir, base_name)

    # Ground truth classes
    gt_classes = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                cls_id = int(float(line.split()[0]))
                gt_classes.append(cls_id)
    if not gt_classes:
        continue  # Skip if no labels

   
    result = model.predict(img_path, conf=0.25, iou=0.5, verbose=False)[0]
    pred_classes = result.boxes.cls.cpu().numpy().astype(int) if result.boxes else []

    # Match predicted length to ground truth length for comparison
    if len(pred_classes) >= len(gt_classes):
        y_true.extend(gt_classes)
        y_pred.extend(pred_classes[:len(gt_classes)])
    else:
        y_true.extend(gt_classes[:len(pred_classes)])
        y_pred.extend(pred_classes)

# Print clean classification report
print(classification_report(y_true, y_pred, zero_division=0))


# In[22]:


import os
import glob
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import classification_report

# Load the trained model again
model = YOLO("/kaggle/working/kaggle_yolo_training/coco_mini_4000_run2/weights/best.pt")

val_img_dir = "/kaggle/input/coco-mini-4000/coco-mini-4000/images/val2017"
val_lbl_dir = "/kaggle/input/coco-mini-4000/coco-mini-4000/labels/val2017"

y_true = []
y_pred = []

image_paths = sorted(glob.glob(os.path.join(val_img_dir, "*.jpg")))

for img_path in image_paths:
    base_name = os.path.basename(img_path).replace(".jpg", ".txt")
    label_path = os.path.join(val_lbl_dir, base_name)

    gt_classes = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                cls_id = int(float(line.split()[0]))
                gt_classes.append(cls_id)

    if not gt_classes:
        continue

    result = model.predict(img_path, conf=0.25, iou=0.5, verbose=False)[0]
    pred_classes = result.boxes.cls.cpu().numpy().astype(int) if result.boxes else []

    if len(pred_classes) >= len(gt_classes):
        y_true.extend(gt_classes)
        y_pred.extend(pred_classes[:len(gt_classes)])
    else:
        y_true.extend(gt_classes[:len(pred_classes)])
        y_pred.extend(pred_classes)


# In[84]:


from sklearn.metrics import classification_report

report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

# Save it to file
import json
with open("yolov8_classification_report.json", "w") as f:
    json.dump(report, f, indent=2)


# In[82]:


from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, zero_division=0))


# In[25]:


# COCO class names (0 to 79)
coco_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

from sklearn.metrics import classification_report

# Generate the report dictionary
report_dict = classification_report(y_true, y_pred, zero_division=0, output_dict=True)

# Format results as table
rows = []
for i in range(80):
    cls_name = coco_classes[i]
    metrics = report_dict.get(str(i))
    if metrics:
        rows.append({
            "Class ID": i,
            "Class Name": cls_name,
            "Precision": round(metrics['precision'], 2),
            "Recall": round(metrics['recall'], 2),
            "F1-Score": round(metrics['f1-score'], 2),
            "Support": int(metrics['support'])
        })

import pandas as pd

df_report = pd.DataFrame(rows)
df_report.head(80)  # Show full COCO class-wise report



# In[26]:


get_ipython().system('git clone https://github.com/ultralytics/yolov5')
get_ipython().run_line_magic('cd', 'yolov5')
get_ipython().system('pip install -r requirements.txt')


# In[31]:


get_ipython().system('cp /kaggle/working/coco-mini-4000.yaml /kaggle/working/yolov5/')


# In[32]:


import os
os.environ["WANDB_MODE"] = "disabled" 

get_ipython().system('python train.py    --img 640    --batch 8    --epochs 15    --data coco-mini-4000.yaml    --weights yolov5m.pt    --project yolov5_train    --name coco_mini_4000_run    --exist-ok')


# In[42]:


import pandas as pd

df = pd.read_csv("/kaggle/working/yolov5/yolov5_train/coco_mini_4000_run/results.csv")
df.head()


# In[48]:


import pandas as pd


df = pd.read_csv("/kaggle/working/yolov5/yolov5_train/coco_mini_4000_run/results.csv")
df.columns = df.columns.str.strip()  # 🔧 removes leading spaces from column names


# In[49]:


import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))

# 🔸 Training Losses
plt.subplot(1, 2, 1)
plt.plot(df['epoch'], df['train/box_loss'], label='Box Loss')
plt.plot(df['train/obj_loss'], label='Obj Loss')
plt.plot(df['train/cls_loss'], label='Cls Loss')
plt.title("Training Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# 🔸 Evaluation Metrics
plt.subplot(1, 2, 2)
plt.plot(df['metrics/precision'], label='Precision')
plt.plot(df['metrics/recall'], label='Recall')
plt.plot(df['metrics/mAP_0.5'], label='mAP@0.5')
plt.plot(df['metrics/mAP_0.5:0.95'], label='mAP@0.5:0.95')
plt.title("Evaluation Metrics")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()

plt.tight_layout()
plt.show()


# In[51]:


get_ipython().system('find /kaggle/working -name best.pt')


# In[55]:


import os
import glob
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report
from PIL import Image
from torchvision import transforms


model = torch.hub.load("ultralytics/yolov5", "custom", path="/kaggle/working/yolov5/yolov5_train/coco_mini_4000_run/weights/best.pt", force_reload=True)
model.conf = 0.25  # confidence threshold

# Paths
val_img_dir = "/kaggle/input/coco-mini-4000/coco-mini-4000/images/val2017"
val_lbl_dir = "/kaggle/input/coco-mini-4000/coco-mini-4000/labels/val2017"

y_true = []
y_pred = []

image_paths = sorted(glob.glob(f"{val_img_dir}/*.jpg"))

for img_path in image_paths:
    label_path = os.path.join(val_lbl_dir, os.path.basename(img_path).replace(".jpg", ".txt"))

    # Read ground truth labels
    gt_classes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                cls_id = int(float(line.split()[0]))
                gt_classes.append(cls_id)
    if not gt_classes:
        continue

    # Inference
    results = model(img_path, size=640)
    pred_classes = results.xywh[0][:, -1].cpu().numpy().astype(int) if len(results.xywh[0]) else []

    # Match length of predictions with ground truth
    if len(pred_classes) >= len(gt_classes):
        y_true.extend(gt_classes)
        y_pred.extend(pred_classes[:len(gt_classes)])
    else:
        y_true.extend(gt_classes[:len(pred_classes)])
        y_pred.extend(pred_classes)

# Final classification report
print(classification_report(y_true, y_pred, zero_division=0))


# In[56]:


get_ipython().system('git clone https://github.com/ultralytics/yolov3')
get_ipython().run_line_magic('cd', 'yolov3')
get_ipython().system('pip install -r requirements.txt')


# In[58]:


get_ipython().system('cp /kaggle/working/coco-mini-4000.yaml /kaggle/working/yolov5/yolov3/')


# In[59]:


get_ipython().system("python train.py    --img 640    --batch 8    --epochs 15    --data coco-mini-4000.yaml    --cfg models/yolov3.yaml    --weights ''    --name coco_mini_yolov3    --project yolov3_train    --exist-ok")


# In[60]:


get_ipython().system('find /kaggle/working -name results.csv')


# In[64]:


import pandas as pd

df = pd.read_csv("/kaggle/working/yolov5/yolov3/yolov3_train/coco_mini_yolov3/results.csv")
df.columns = df.columns.str.strip()  
print(df.columns.tolist())


# In[69]:


import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("/kaggle/working/yolov5/yolov3/yolov3_train/coco_mini_yolov3/results.csv")  # replace with actual path
df.columns = df.columns.str.strip()

# 🔹 Training Loss Plot
plt.figure(figsize=(8, 5))
plt.plot(df['epoch'], df['train/box_loss'], label='Box Loss')
plt.plot(df['train/obj_loss'], label='Obj Loss')
plt.plot(df['train/cls_loss'], label='Cls Loss')
plt.title("YOLOv3 Training Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("yolov3_training_losses.png")  # 📸 Save loss graph
plt.close()

# 🔹 Evaluation Metrics Plot
plt.figure(figsize=(8, 5))
plt.plot(df['epoch'], df['metrics/precision'], label='Precision')
plt.plot(df['metrics/recall'], label='Recall')
plt.plot(df['metrics/mAP_0.5'], label='mAP@0.5')
plt.plot(df['metrics/mAP_0.5:0.95'], label='mAP@0.5:0.95')
plt.title("YOLOv3 Evaluation Metrics")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.savefig("yolov3_eval_metrics.png")  # 📸 Save metrics graph
plt.close()


# In[70]:


from IPython.display import Image, display
display(Image(filename="yolov3_training_losses.png"))
display(Image(filename="yolov3_eval_metrics.png"))


# In[73]:


import os
import glob
import torch
import numpy as np
from sklearn.metrics import classification_report


model = torch.hub.load("ultralytics/yolov3", "custom", path="/kaggle/working/yolov5/yolov3/yolov3_train/coco_mini_yolov3/weights/best.pt", force_reload=True)
model.conf = 0.25  # confidence threshold

# Dataset paths
val_img_dir = "/kaggle/input/coco-mini-4000/coco-mini-4000/images/val2017"
val_lbl_dir = "/kaggle/input/coco-mini-4000/coco-mini-4000/labels/val2017"

y_true = []
y_pred = []

# Load validation images
image_paths = sorted(glob.glob(os.path.join(val_img_dir, "*.jpg")))

for img_path in image_paths:
    label_path = os.path.join(val_lbl_dir, os.path.basename(img_path).replace(".jpg", ".txt"))

    # Ground truth labels
    gt_classes = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                cls_id = int(float(line.strip().split()[0]))
                gt_classes.append(cls_id)
    if not gt_classes:
        continue

    # Run YOLOv3 inference
    results = model(img_path, size=640)
    pred_classes = results.xywh[0][:, -1].cpu().numpy().astype(int) if len(results.xywh[0]) else []

    # Match predicted vs ground truth
    if len(pred_classes) >= len(gt_classes):
        y_true.extend(gt_classes)
        y_pred.extend(pred_classes[:len(gt_classes)])
    else:
        y_true.extend(gt_classes[:len(pred_classes)])
        y_pred.extend(pred_classes)

# Final classification report
print(classification_report(y_true, y_pred, zero_division=0))


# In[75]:


get_ipython().system('find /kaggle/working -name results.csv')


# In[106]:


import pandas as pd
import matplotlib.pyplot as plt


paths = {
    "YOLOv3": "/kaggle/working/yolov5/yolov3/yolov3_train/coco_mini_yolov3/results.csv",
    "YOLOv5": "/kaggle/working/yolov5/yolov5_train/coco_mini_4000_run/results.csv",
    "YOLOv8": "/kaggle/working/kaggle_yolo_training/coco_mini_4000_run2/results.csv"
}


summary = []
for model, path in paths.items():
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        last_row = df.iloc[-1]
        summary.append({
            "Model": model,
            "Precision": round(last_row.get("metrics/precision", 0), 4),
            "Recall": round(last_row.get("metrics/recall", 0), 4),
            "mAP@0.5": round(last_row.get("metrics/mAP_0.5", 0), 4),
            "mAP@0.5:0.95": round(last_row.get("metrics/mAP_0.5:0.95", 0), 4)
        })
    except Exception as e:
        print(f" Could not load {model}: {e}")

# Create and plot comparison DataFrame
comparison_df = pd.DataFrame(summary)

if not comparison_df.empty:
    ax = comparison_df.set_index("Model").plot(kind="bar", figsize=(10, 6))
    plt.title("YOLOv3 vs YOLOv5 vs YOLOv8 - Performance Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.grid(axis="y")
    plt.tight_layout()


    plt.savefig("yolo_comparison.png")
    plt.show()
else:
    print("No data available to plot. Please verify your file paths.")


# In[107]:


from IPython.display import Image, display
display(Image("yolo_comparison.png"))


# In[114]:


import pandas as pd
import matplotlib.pyplot as plt


# Format: [Precision, Recall, mAP@0.5, mAP@0.5:0.95]
manual_metrics = {
    "YOLOv3": [0.406, 0.083, 0.0532, 0.0251],  
    "YOLOv5": [0.675, 0.560, 0.609, 0.416],
    "YOLOv8": [0.6551, 0.5472, 0.5871, 0.4282]
}

# Create summary DataFrame
summary = []
for model, values in manual_metrics.items():
    summary.append({
        "Model": model,
        "Precision": values[0],
        "Recall": values[1],
        "mAP@0.5": values[2],
        "mAP@0.5:0.95": values[3]
    })

comparison_df = pd.DataFrame(summary)

# Plotting
ax = comparison_df.set_index("Model").plot(kind="bar", figsize=(10, 6))
plt.title("YOLOv3 vs YOLOv5 vs YOLOv8 - Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.ylim(0, 1.0)
plt.grid(axis="y")
plt.tight_layout()

# Save and show
plt.savefig("yolo_comparison_manual.png")
plt.show()


# In[115]:


from IPython.display import Image, display

# Display the saved image
display(Image("yolo_comparison_manual.png"))

