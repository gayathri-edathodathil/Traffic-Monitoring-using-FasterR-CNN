import os
import json
import torch
from PIL import Image
import pandas as pd
from torchvision.transforms import v2 as T


class BDDDataset(torch.utils.data.Dataset):

    def __init__(self, root, annotation_file, transforms=None):
        self.root = root

        with open(annotation_file) as f:
            self.annotations = json.load(f)

        self.valid = []

        for item in self.annotations:
            if 'labels' in item:
                self.valid.append(item)

        self.class_map = {
            "person": 1,
            "rider": 2,
            "car": 3,
            "truck": 4,
            "bus": 5,
            "train": 6,
            "motor": 7,
            "bike": 8,
            "traffic light": 9,
            "traffic sign": 10
        }
        self.transforms = transforms

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, idx):

        item = self.valid[idx]

        img_path = os.path.join(self.root, item['name'])
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []

        for obj in item['labels']:

            if obj['category'] not in self.class_map:
                continue

            box = obj['box2d']

            xmin = box['x1']
            ymin = box['y1']
            xmax = box['x2']
            ymax = box['y2']

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_map[obj['category']])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        if self.transforms:
            img, target = self.transforms(img,target)

        return img, target