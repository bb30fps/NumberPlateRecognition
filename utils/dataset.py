import os
import cv2
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings

class NumberPlateDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, chars, transforms=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transforms = transforms
        self.char_to_idx = {char: idx for idx, char in enumerate(chars)}
        self.annotations = self._parse_xmls()

    def _parse_xmls(self):
        annotations = []
        for xml_file in os.listdir(self.annotation_dir):
            try:
                tree = ET.parse(os.path.join(self.annotation_dir, xml_file))
                root = tree.getroot()
                
                filename = root.find('filename').text
                img_path = os.path.join(self.image_dir, filename)
                
                if not os.path.exists(img_path):
                    warnings.warn(f"Missing image {filename}")
                    continue
                    
                for obj in root.findall('object'):
                    plate = obj.find('name').text.upper()
                    bbox = obj.find('bndbox')
                    annotations.append({
                        'filename': filename,
                        'plate': plate,
                        'bbox': [
                            int(bbox.find('xmin').text),
                            int(bbox.find('ymin').text),
                            int(bbox.find('xmax').text),
                            int(bbox.find('ymax').text)
                        ]
                    })
            except Exception as e:
                print(f"Skipping {xml_file}: {str(e)}")
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.image_dir, annotation['filename'])
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        xmin, ymin, xmax, ymax = annotation['bbox']
        image = image[ymin:ymax, xmin:xmax]
        
        label = annotation['plate']
        encoded_label = torch.tensor(
            [self.char_to_idx[c] for c in label], 
            dtype=torch.long
        )

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']

        return image, encoded_label
