from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import requests
from PIL import Image
import io

app = FastAPI()

@app.post("/AI")
async def analyze_ootd(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))

        # Segmentation으로 의류 리스트 추출
        clothing_items = run_segmentation(img) #output : setmentation된 각각의 Image

        # 각 의류에 대해 classification 수행 및 결과 전송
        for clothing_img in clothing_items:
            result = run_classification(clothing_img)

            data = {
                "imageSrc": clothing_img,
                "type": result["type"],
                "detail": result["detail"],
                "print": result["print"],
                "texture": result["texture"],
                "style": result["style"]
            }

            url = "BackendUrl" #url 임의로 설정됨 
            response = requests.post(url, json=data)

            if response.status_code != 200:
                raise ValueError(f"Failed to send result")

        return JSONResponse(content={"message": "모든 의류 분석 결과 전송 완료"}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)