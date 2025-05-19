from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import io
import os

# 모델과 전처리기 (전역 1회만 로드)
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
model.eval()

# 대상 클래스만 사용
CLASS_COLOR_MAP = {
    4: (0, 128, 255),    # Upper-clothes
    5: (255, 0, 128),    # Skirt
    6: (0, 255, 0),      # Pants
    7: (255, 128, 0),    # Dress
}

def run_segmentation(image: Image.Image):
    image_rgb = image.convert("RGB")  # 모델에는 RGB만 전달해야 함
    inputs = processor(images=image_rgb, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    upsampled_logits = F.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()

    segmented_images = []

    for cls, color in CLASS_COLOR_MAP.items():
        mask = (pred_seg == cls).astype(np.uint8) * 255
        if np.sum(mask) == 0:
            continue  # 해당 클래스 없음

        # 마스크를 알파 채널로 적용하기 위해 원본은 RGBA로
        image_rgba = image.convert("RGBA")
        alpha_mask = Image.fromarray(mask).convert("L")
        image_rgba.putalpha(alpha_mask)

        # bbox로 크롭 (optional)
        bbox = alpha_mask.getbbox()
        if bbox:
            image_rgba = image_rgba.crop(bbox)
        """
        # 저장 및 반환 -> 테스트 이후 주석처리 
        filename = f"seg_cropped_cls{cls}.png"
        image_rgba.save(filename)
        print(f"[✓] Saved cropped: {filename}")
        """
        buffer = io.BytesIO()
        image_rgba.save(buffer, format="PNG")
        buffer.seek(0)
        segmented_images.append(buffer)

    return segmented_images