from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import io

# 파이썬용 원본 모델 ID
MODEL_ID = "mattmdjaga/segformer_b2_clothes"

# 모델 로드
processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_ID)
model.eval()

# 1. 남길 '패션 아이템' ID 정의 (신체 부위 제외)
# 제외된 것: 0(배경), 2(머리카락), 11(얼굴), 12~15(팔/다리)
TARGET_CLOTH_IDS = {
    1,  # Hat (모자)
    3,  # Sunglasses (선글라스)
    4,  # Upper-clothes (상의)
    5,  # Skirt (치마)
    6,  # Pants (바지)
    7,  # Dress (원피스)
    8,  # Belt (벨트)
    9,  # Left-shoe (신발 - 병합됨)
    10, # Right-shoe (신발 - 병합 로직용)
    16, # Bag (가방)
    17  # Scarf (스카프)
}

# 2. 신발 병합 매핑 (Right-shoe -> Left-shoe로 통합)
MERGE_MAPPING = {
    10: 9 
}

def run_segmentation(image: Image.Image):
    image_rgb = image.convert("RGB")
    
    # 전처리 (원본 비율 유지)
    inputs = processor(images=image_rgb, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    
    # 마스크 업샘플링
    upsampled_logits = F.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()

    segmented_images = [] 

    # 전체 클래스 탐색
    for cls_id in range(1, 18):
        
        # [핵심 수정] 우리가 원하는 '옷/악세서리'가 아니면 즉시 건너뜀 (얼굴, 팔, 다리 등 제거)
        if cls_id not in TARGET_CLOTH_IDS:
            continue

        # 병합될 대상(오른발 10)은 건너뜀 (9번 처리할 때 합쳐짐)
        if cls_id in MERGE_MAPPING:
            continue
            
        # 신발(9)인 경우: 9번 + 10번 합치기
        if cls_id == 9: 
            mask = np.logical_or(pred_seg == 9, pred_seg == 10).astype(np.uint8) * 255
        # 그 외: 해당 클래스 마스크만 사용
        else:
            mask = (pred_seg == cls_id).astype(np.uint8) * 255

        # 해당 아이템이 사진에 없으면 패스
        if np.sum(mask) == 0:
            continue

        # --- 이미지 저장 및 크롭 ---
        image_rgba = image.convert("RGBA")
        alpha_mask = Image.fromarray(mask).convert("L")
        image_rgba.putalpha(alpha_mask)

        # BBox로 타이트하게 크롭
        bbox = alpha_mask.getbbox()
        if bbox:
            image_rgba = image_rgba.crop(bbox)

        buffer = io.BytesIO()
        image_rgba.save(buffer, format="PNG")
        buffer.seek(0)
        
        # 순수 의류 이미지 버퍼만 리스트에 추가
        segmented_images.append(buffer)

    return segmented_images