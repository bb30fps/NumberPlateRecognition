import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.dataset import NumberPlateDataset
from models.model import PlateRecognitionModel
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml

# Load config
with open("utils/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Data Augmentation
transform = A.Compose([
    A.Resize(*config['image_size']),
    A.Normalize(),
    ToTensorV2()
])

# Dataset
dataset = NumberPlateDataset(
    image_dir="data/images",
    annotation_dir="data/annotations",
    transforms=transform
)
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

# Model
model = PlateRecognitionModel(num_chars=len(config['chars']))
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
criterion = nn.CTCLoss()

# Training loop
for epoch in range(config['num_epochs']):
    for images, bboxes, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Save model
torch.save(model.state_dict(), "models/number_plate_model.pth")
