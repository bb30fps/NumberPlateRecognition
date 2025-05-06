import torch
from torch.utils.data import DataLoader, Subset
from utils.dataset import NumberPlateDataset
from models.model import PlateRecognitionModel
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
import warnings

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- Load Config ----
with open("utils/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# ---- Collate Function ----
def collate_fn(batch):
    """Handle variable-length sequence labels"""
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

# ---- Data Preparation with Validation Split ----
transform = A.Compose([
    A.Resize(*config['image_size']),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Load full dataset
full_dataset = NumberPlateDataset(
    image_dir="data/images",
    annotation_dir="data/annotations",
    chars=config['chars'],
    transforms=transform
)

# Split dataset
train_indices, val_indices = train_test_split(
    list(range(len(full_dataset))),
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# Create subset datasets
train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    collate_fn=collate_fn,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config['batch_size'],
    collate_fn=collate_fn,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# ---- Class Balancing ----
# Calculate character frequencies for loss weighting
char_counts = torch.zeros(len(config['chars']))
for idx in train_indices:
    plate_text = full_dataset.annotations[idx]['plate']
    for c in plate_text:
        char_counts[full_dataset.char_to_idx[c]] += 1

# Add smoothing to prevent division by zero
class_weights = 1.0 / (char_counts + 1e-6)  
class_weights /= class_weights.sum()  # Normalize

# ---- Model Setup with Weighted Loss ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PlateRecognitionModel(num_chars=len(config['chars'])).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
criterion = torch.nn.CTCLoss(
    blank=len(config['chars']),
    weight=class_weights.to(device)
)

# ---- Enhanced Training Loop with Validation ----
best_val_loss = float('inf')
for epoch in range(config['num_epochs']):
    # Training phase
    model.train()
    train_loss = 0.0
    
    for batch_idx, (images, labels, label_lengths) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        log_probs = torch.nn.functional.log_softmax(outputs, dim=2)
        
        input_lengths = torch.full(
            size=(images.size(0),),
            fill_value=outputs.size(0),
            dtype=torch.long
        ).to(device)
        
        loss = criterion(
            log_probs,
            labels,
            input_lengths,
            label_lengths
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        
        train_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels, label_lengths in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)
            
            outputs = model(images)
            log_probs = torch.nn.functional.log_softmax(outputs, dim=2)
            
            input_lengths = torch.full(
                size=(images.size(0),),
                fill_value=outputs.size(0),
                dtype=torch.long
            ).to(device)
            
            loss = criterion(
                log_probs,
                labels,
                input_lengths,
                label_lengths
            )
            val_loss += loss.item()

    # Epoch statistics
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1} Complete")
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'chars': config['chars'],
            'val_loss': avg_val_loss
        }, "models/number_plate_model.pth")
        print(f"New best model saved with val loss {avg_val_loss:.4f}")

print("Training complete! Best model saved to models/number_plate_model.pth")
