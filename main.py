import io
import os
import json
import sys
import httpx
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import asyncio

sys.path.append(os.path.join(os.path.dirname(__file__), "ootd-segmentation"))
sys.path.append(os.path.join(os.path.dirname(__file__), "ootd-classification"))

from run_segmentation import run_segmentation
from run_classification import run_classification

app = FastAPI()

@app.post("/ai")
async def analyze_ootd(
    image: UploadFile = File(...),
    user_id: int = Form(...),
    post_id: int = Form(...)
):
    try:
        print(f"user_id: {user_id}, post_id: {post_id}")
        sys.stdout.flush()

        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        img.load()

        # segmentation
        try:
            clothing_items = run_segmentation(img)
            print(f"{len(clothing_items)} clothing items segmented")
        except Exception as e:
            print(f"Segmentation failed: {e}")
            sys.stdout.flush()
            return JSONResponse(content={"error": f"Segmentation failed: {e}"}, status_code=500)

        backend_url = os.getenv("BACKEND_URL", "http://localhost:8080")

        async with httpx.AsyncClient(timeout=None) as client:
            tasks = []

            for idx, clothing_img in enumerate(clothing_items):
                try:
                    print(f"Running classification for item {idx+1}")
                    sys.stdout.flush()

                    clothing_img.seek(0)
                    pil_image = Image.open(clothing_img)

                    result = run_classification(pil_image)

                    data = {
                        "user_id": user_id,
                        "post_id": post_id,
                        "type": result.get("type"),
                        "detail": result.get("detail"),
                        "print": result.get("print"),
                        "texture": result.get("texture"),
                        "style": result.get("style")
                    }

                    clothing_img_bytes = io.BytesIO()
                    pil_image.save(clothing_img_bytes, format="PNG")
                    clothing_img_bytes.seek(0)

                    files = {
                        "file": ("image.png", clothing_img_bytes, "image/png"),
                        "data": ("data.json", json.dumps(data), "application/json") 
                    }

                    url = f"{backend_url}/clothes"
                    tasks.append(client.post(url, files=files))

                except Exception as e:
                    print(f"Classification failed for item {idx+1}: {e}")
                    sys.stdout.flush()
                    continue

            if tasks:
                await asyncio.gather(*tasks)

        return JSONResponse(content={"message": "분석 완료"}, status_code=200)

    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.stdout.flush()
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8001)