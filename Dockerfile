# Python 3.8.18 기반 이미지 사용
FROM python:3.8.18-slim

# 작업 디렉터리 설정
WORKDIR /app

# 시스템 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    git curl && \
    rm -rf /var/lib/apt/lists/*

# requirements.txt 복사 및 설치
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 프로젝트 전체 복사
COPY . .

# build-time argument: Hugging Face 토큰 (외부에서 주입됨)
ARG HF_TOKEN
ENV HUGGINGFACE_HUB_TOKEN=$HF_TOKEN

# Hugging Face 모델 5개 다운로드
RUN python -c "\
from huggingface_hub import hf_hub_download; \
import os; \
model_files = [ \
    'model_print_best.pth.tar', \
    'model_category_best.pth.tar', \
    'model_style_best.pth.tar', \
    'model_detail_best.pth.tar', \
    'model_texture_best.pth.tar' \
]; \
[ \
    hf_hub_download( \
        repo_id='pupuready/ootd-image-classification', \
        filename=f, \
        local_dir='ootd-classification/checkpoint', \
        repo_type='model', \
        token=os.environ['HUGGINGFACE_HUB_TOKEN'] \
    ) for f in model_files \
]"

# FastAPI 앱 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

