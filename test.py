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

        print(f"[🧵] {len(clothing_items)} clothing items segmented")
        sys.stdout.flush()

        results = []

        for idx, clothing_img in enumerate(clothing_items):
            print(f"[🔍] Running classification for item {idx+1}")
            sys.stdout.flush()
            print(f"[🔍] type(clothing_img): {type(clothing_img)}")
            result = run_classification(clothing_img)

            data = {
                "imageSrc": pil_to_base64(clothing_img),
                "type": result["type"],
                "detail": result["detail"],
                "print": result["print"],
                "texture": result["texture"],
                "style": result["style"]
            }

            print(f"[🧾] Classification result {idx+1}: {data}")
            sys.stdout.flush()

            results.append(data)

        return JSONResponse(content={"message": "분석 완료", "results": results}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
