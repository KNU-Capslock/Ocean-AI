from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import requests
from PIL import Image
import io
import sys
import os
import base64

sys.path.append(os.path.join(os.path.dirname(__file__), "ootd-segmentation"))
sys.path.append(os.path.join(os.path.dirname(__file__), "ootd-classification"))

from run_segmentation import run_segmentation
from run_classification import run_classification
app = FastAPI()

def pil_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

@app.post("/AI")
async def analyze_ootd(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img_bytes = io.BytesIO(contents)
        img = Image.open(img_bytes)
        img.load() 

        clothing_items = run_segmentation(img)

        print(f"{len(clothing_items)} clothing items segmented")
        sys.stdout.flush()

        results = []

        for idx, clothing_img in enumerate(clothing_items):
            print(f"Running classification for item {idx+1}")
            sys.stdout.flush()
            clothing_img = Image.open(clothing_img)
            clothing_img.load()
            clothing_img = clothing_img.convert("RGB")
            result = run_classification(clothing_img)

            data = {
                "type": result["type"],
                "detail": result["detail"],
                "print": result["print"],
                "texture": result["texture"],
                "style": result["style"]
            }
            
            # multipart용 이미지 파일 구성
            clothing_img_bytes = io.BytesIO()
            clothing_img.save(clothing_img_bytes, format="PNG")
            clothing_img_bytes.seek(0)

            files = {
                "image_file": ("image.png", clothing_img_bytes, "image/png")
            }

            print(f"Classification result {idx+1}: {data}")
            sys.stdout.flush()
            results.append(data)
            
            # 백엔드 전송
            url = "http://localhost:8000/result"  # 벡엔드 URL로 변경
            response = requests.post(url, files=files, data=data)
        return JSONResponse(content={"message": "분석 완료"}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
