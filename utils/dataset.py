import os
import cv2
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

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
            tree = ET.parse(os.path.join(self.annotation_dir, xml_file))
            root = tree.getroot()
            
            filename = root.find('filename').text
            plate = root.find('object/name').text.upper()  # Ensure uppercase
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
        
        # Read and crop image to license plate region
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        xmin, ymin, xmax, ymax = annotation['bbox']
        image = image[ymin:ymax, xmin:xmax]  # Crop to plate
        
        # Encode text label to numerical indices
        label = annotation['plate']
        encoded_label = [self.char_to_idx[c] for c in label]
        encoded_label = torch.tensor(encoded_label, dtype=torch.long)

        # Apply transforms to cropped plate image
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']

        return image, encoded_label
