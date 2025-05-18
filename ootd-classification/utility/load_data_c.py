import torch.utils.data as data
import json
import os
from utility.util import *

class load_data(data.Dataset):
    def __init__(self, attribute, phase='train', num_classes = 0):
        self.phase = phase
        self.attribute = attribute
        self.root = '../data/kfashion_{}'.format(self.attribute)
        self.img_list = []
        self.get_anno()
        self.num_classes = num_classes

    def get_anno(self):
        list_path = os.path.join(self.root, '{}_anno_{}_final.json'.format(self.phase, self.attribute))
        self.img_list = json.load(open(list_path, 'r'))
        self.cat2idx = json.load(open(os.path.join(self.root, 'category_{}_final.json').format(self.attribute), 'r'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        return self.get(item)

    def get(self, item):
        filename = item['file_name']
        # 파일 경로의 마지막 파일명을 '1.jpg'로 변경
        directory = os.path.dirname(filename)  # 폴더 경로 추출
        filename = os.path.join(directory, "1.jpg")  # 새로운 파일 경로 설정

        if self.attribute == 'category':
            label = [self.cat2idx[item['labels']]]
        else:
            label = [self.cat2idx[x] for x in item['labels']]

        img = Image.open(filename).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        target = np.zeros(self.num_classes, np.float32) - 1
        target[label] = 1
        return (img, filename), target
