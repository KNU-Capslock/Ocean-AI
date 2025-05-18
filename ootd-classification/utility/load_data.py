import torch.utils.data as data
import json
import os
import pickle
from utility.util import *
from PIL import Image
import numpy as np

class load_data(data.Dataset):
    def __init__(self, root, phase='train', inp_name=None):
        self.root = root
        self.phase = phase
        self.img_list = []
        self.get_anno()
        self.num_classes = len(self.cat2idx)

        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)

        self.inp_name = inp_name

    def get_anno(self):
        list_path = os.path.join(self.root, '{}_anno_custom_final_0.json'.format(self.phase))
        self.img_list = json.load(open(list_path, 'r'))
        self.cat2idx = json.load(open(os.path.join(self.root, 'category_custom_final.json'), 'r'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        return self.get(item)

    def get(self, item):
        filename = item['file_name']  # 원본 파일 경로
        labels = sorted(item['labels'])

        # 파일 경로의 마지막 파일명을 '1.jpg'로 변경
        directory = os.path.dirname(filename)  # 폴더 경로 추출
        filename = os.path.join(directory, "1.jpg")  # 새로운 파일 경로 설정

        # 이미지 로드
        img = Image.open(filename).convert('RGB')

        # 변환 적용
        if self.transform is not None:
            img = self.transform(img)

        # 타겟 벡터 생성
        target = np.zeros(self.num_classes, np.float32) - 1
        target[labels] = 1

        return (img, filename, self.inp), target