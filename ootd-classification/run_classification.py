import os
import json
import torch
import pickle
import torchvision.transforms as transforms
from PIL import Image
from utility.resnest import resnest50d
import sys

# 경로 설정
BASE_DIR = os.path.dirname(__file__)
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoint")
CLASS_MAP_DIR = os.path.join(BASE_DIR, "class")
STYLE_ADJ_PATH = os.path.join(BASE_DIR, "utility", "kfashion_style", "custom_adj_final.pkl")
STYLE_WORD_PATH = os.path.join(BASE_DIR, "utility", "kfashion_style", "custom_glove_word2vec_final.pkl")

# 클래스 이름 매핑 캐시
type_to_classmap = {
    "type": "category",
    "detail": "detail",
    "print": "print",
    "texture": "texture",
    "style": "style"
}

# 전처리 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# GCN용 입력 벡터 로드 
with open(STYLE_WORD_PATH, 'rb') as f:
    style_inp = torch.FloatTensor(pickle.load(f)).unsqueeze(0)

def run_classification(image: Image.Image) -> dict:
    image = image.convert("RGB")
    img_tensor = transform(image).unsqueeze(0)  # shape: [1, 3, 224, 224]
    print("[👕] run_classification.py 실행됨")
    sys.stdout.flush()
    result = {}

    for attr in ["type", "detail", "print", "texture", "style"]:
        class_type = type_to_classmap[attr]

        # 클래스 매핑 불러오기
        with open(os.path.join(CLASS_MAP_DIR, f"category_{class_type}.json"), "r", encoding="utf-8") as f:
            label_map = json.load(f)
        idx_to_label = {v: k for k, v in label_map.items()}
        num_classes = len(idx_to_label)

        # style 모델 로드
        if attr == "style":
            from utility.ml_gcn import gcn_resnet101
            model = gcn_resnet101(num_classes=num_classes, t=0.03, adj_file=STYLE_ADJ_PATH)
        else:
            model = resnest50d(pretrained=False, nc=num_classes)

        # 가중치 로드
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_{class_type}_best.pth.tar")
        checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        with torch.no_grad():
            output = model(img_tensor, style_inp) if attr == "style" else model(img_tensor)
            pred = torch.argmax(output, dim=1).item()
            result[attr] = idx_to_label[pred]

    return result
