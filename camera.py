# 终极修复版！！！！！
import cv2
import torch
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# ===================== 你的文件名是 yolo26l.pt =====================

MODEL_PATH = Path(__file__).resolve().parent / "yolo26l.pt"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"未找到模型文件: {MODEL_PATH}")

# 关键修复：weights_only=False
ckpt = torch.load(str(MODEL_PATH), map_location='cpu', weights_only=False)
model = ckpt['model'].float()
model.eval()

# COCO 类别
CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# 打开摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("摄像头打开失败！")
    exit()

print("✅ 摄像头已启动！按 q 退出")

# NMS
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    if prediction.dtype == torch.float16:
        prediction = prediction.float()
    xc = prediction[..., 4] > conf_thres
    output = []
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue
        box, conf, cls = x.split((4, 1, 1), 1)
        j = torch.where(conf > conf_thres)[0]
        x = torch.cat((box, conf, cls), 1)[j]
        output.append(x[x[:, 4].argsort(descending=True)])
    return output

# 主循环
while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = frame.copy()
    img = cv2.resize(img, (640, 640))
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float() / 255.0
    img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img)[0]

    pred = non_max_suppression(pred, 0.25, 0.45)[0]

    if len(pred):
        for *xyxy, conf, cls in pred:
            x1, y1, x2, y2 = map(int, xyxy)
            label = f"{CLASS_NAMES[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLO 本地摄像头识别", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()