from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import io

# 파이썬용 원본 모델 ID
MODEL_ID = "mattmdjaga/segformer_b2_clothes"

# 모델 로드 (앱처럼 강제 리사이징 하지 않음 -> 원본 비율 유지로 정확도 향상)
processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_ID)
model.eval()

# 신발 병합 매핑 (오른발(10) -> 왼발(9)로 통합)
# 내부적으로 마스크를 합칠 때만 사용합니다.
MERGE_MAPPING = {
    10: 9 
}

def run_segmentation(image: Image.Image):
    image_rgb = image.convert("RGB")
    
    # 전처리
    inputs = processor(images=image_rgb, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    
    # 원본 크기로 마스크 복원
    upsampled_logits = F.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()

    segmented_images = [] # 반환값: 잘린 이미지(bytes)들의 리스트

    # 리드미 기준 1~17번 클래스 전체 탐색
    # (기존엔 4~7번만 있어서 모자, 가방, 신발이 누락되었음)
    for cls_id in range(1, 18):
        # 병합될 대상(오른발 10)은 건너뜀
        if cls_id in MERGE_MAPPING:
            continue
            
        # 신발(9)인 경우: 9번(왼발) + 10번(오른발) 마스크 합치기
        if cls_id == 9: 
            mask = np.logical_or(pred_seg == 9, pred_seg == 10).astype(np.uint8) * 255
        # 그 외: 해당 클래스 마스크만 사용
        else:
            mask = (pred_seg == cls_id).astype(np.uint8) * 255

        # 해당 객체가 없으면 패스
        if np.sum(mask) == 0:
            continue

        # --- 이미지 저장 및 크롭 로직 ---
        image_rgba = image.convert("RGBA")
        alpha_mask = Image.fromarray(mask).convert("L")
        image_rgba.putalpha(alpha_mask)

        # 의류 영역만큼만 잘라내기 (BBox)
        bbox = alpha_mask.getbbox()
        if bbox:
            image_rgba = image_rgba.crop(bbox)

        buffer = io.BytesIO()
        image_rgba.save(buffer, format="PNG")
        buffer.seek(0)
        
        # 라벨 정보 없이 이미지만 리스트에 추가 (main.py와 호환)
        segmented_images.append(buffer)

    return segmented_images