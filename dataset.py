import os
import cv2
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset

class NumberPlateDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transforms=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transforms = transforms
        self.image_files = os.listdir(image_dir)
        self.annotations = self._parse_xmls()

    def _parse_xmls(self):
        annotations = []
        for xml_file in os.listdir(self.annotation_dir):
            tree = ET.parse(os.path.join(self.annotation_dir, xml_file))
            root = tree.getroot()
            filename = root.find('filename').text
            plate = root.find('object/name').text
            xmin = int(root.find('object/bndbox/xmin').text)
            ymin = int(root.find('object/bndbox/ymin').text)
            xmax = int(root.find('object/bndbox/xmax').text)
            ymax = int(root.find('object/bndbox/ymax').text)
            annotations.append({
                'filename': filename,
                'plate': plate,
                'bbox': [xmin, ymin, xmax, ymax]
            })
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.image_dir, annotation['filename'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bbox = annotation['bbox']
        label = annotation['plate']

        if self.transforms:
            transformed = self.transforms(image=image, bboxes=[bbox], labels=[label])
            image = transformed['image']
            bbox = transformed['bboxes'][0]
            label = transformed['labels'][0]

        return image, bbox, label
