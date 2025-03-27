import torch
from torch.utils.data import DataLoader
from utils.dataset import NumberPlateDataset
from models.model import PlateRecognitionModel
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml
import numpy as np

# ---- 1. Load Config ----
with open("utils/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# ---- 2. Collate Function ----
def collate_fn(batch):
    images = []
    labels = []
    label_lengths = []
    
    for img, label in batch:
        images.append(img)
        labels.append(label)
        label_lengths.append(len(label))
        
    images = torch.stack(images, dim=0)
    labels = torch.cat(labels, dim=0)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)
    
    return images, labels, label_lengths

# ---- 3. Data Preparation ----
transform = A.Compose([
    A.Resize(*config['image_size']),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

dataset = NumberPlateDataset(
    image_dir="data/images",
    annotation_dir="data/annotations",
    chars=config['chars'],
    transforms=transform
)

dataloader = DataLoader(
    dataset,
    batch_size=config['batch_size'],
    collate_fn=collate_fn,
    shuffle=True,
    num_workers=2
)

# ---- 4. Model Setup ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PlateRecognitionModel(num_chars=len(config['chars'])).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
criterion = torch.nn.CTCLoss(blank=len(config['chars']))  # Blank index

# ---- 5. Training Loop ----
for epoch in range(config['num_epochs']):
    epoch_loss = 0
    model.train()
    
    for batch_idx, (images, labels, label_lengths) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)  # Shape: (seq_len, batch_size, num_chars+1)
        log_probs = torch.nn.functional.log_softmax(outputs, dim=2)
        
        # Input lengths (all sequences are same length)
        input_lengths = torch.full(
            size=(images.size(0),),
            fill_value=outputs.size(0),  # Sequence length from model
            dtype=torch.long
        ).to(device)
        
        # Compute loss
        loss = criterion(
            log_probs,          # (T, N, C)
            labels,             # (N*S)
            input_lengths,      # (N)
            label_lengths       # (N)
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")
    
    # Epoch statistics
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f}")

# ---- 6. Save Model ----
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
    'chars': config['chars']
}, "models/number_plate_model.pth")

print("Training complete! Model saved.")
