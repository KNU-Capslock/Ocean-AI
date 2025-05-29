from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import requests
from PIL import Image
import io
import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), "ootd-segmentation"))
sys.path.append(os.path.join(os.path.dirname(__file__), "ootd-classification"))

from run_segmentation import run_segmentation
from run_classification import run_classification
app = FastAPI()


@app.post("/ai")
async def analyze_ootd(image: UploadFile = File(...), user_id: int = Form(...)):
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

            result = run_classification(clothing_img)

            data = {
                "user_id" : user_id,
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
                "file": ("image.png", clothing_img_bytes, "image/png"),
                "data": (None, json.dumps(data), "application/json")
            }

            print(f"Classification result {idx+1}: {data}")
            sys.stdout.flush()
            results.append(data)
            
            # 백엔드 전송
            url = "http://localhost:8080/clothes"  # 벡엔드 URL로 변경
            response = requests.post(url, files=files)
        return JSONResponse(content={"message": "분석 완료"}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8001)