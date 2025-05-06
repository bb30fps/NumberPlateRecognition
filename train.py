import torch
from torch.utils.data import DataLoader, Subset
from utils.dataset import NumberPlateDataset
from models.model import PlateRecognitionModel
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml
import os
import sys
from sklearn.model_selection import train_test_split
import torch.multiprocessing
import numpy as np
from visualization import TrainingVisualizer

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def calculate_accuracy(model, loader, device, chars):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, label_lengths in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 2)
            
            pred_strs = decode_predictions(preds, chars)
            true_strs = decode_predictions(labels, chars)
            
            for pred, true in zip(pred_strs, true_strs):
                if pred == true:
                    correct += 1
                total += 1
                    
    return 100 * correct / total if total > 0 else 0

def decode_predictions(preds, chars):
    blank_idx = len(chars)
    sequences = []
    
    for pred in preds.permute(1, 0):
        chars_pred = []
        prev_char = blank_idx
        
        for idx in pred:
            if idx != prev_char and idx != blank_idx:
                chars_pred.append(chars[idx])
            prev_char = idx
            
        sequences.append(''.join(chars_pred))
        
    return sequences

def main():
    # Load config
    with open("utils/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize visualizer
    visualizer = TrainingVisualizer()
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Data preparation
    transform = A.Compose([
        A.Resize(*config['image_size']),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    full_dataset = NumberPlateDataset(
        image_dir="data/images",
        annotation_dir="data/annotations",
        chars=config['chars'],
        transforms=transform
    )

    # Dataset split
    train_indices, val_indices = train_test_split(
        list(range(len(full_dataset))),
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    # DataLoader settings
    num_workers = 0 if os.name == 'nt' else 2
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        Subset(full_dataset, train_indices),
        batch_size=config['batch_size'],
        collate_fn=lambda b: (torch.stack([item[0] for item in b]), 
                             torch.cat([item[1] for item in b]),
                             torch.tensor([len(item[1]) for item in b])),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        Subset(full_dataset, val_indices),
        batch_size=config['batch_size'],
        collate_fn=lambda b: (torch.stack([item[0] for item in b]), 
                             torch.cat([item[1] for item in b]),
                             torch.tensor([len(item[1]) for item in b])),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlateRecognitionModel(num_chars=len(config['chars'])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.CTCLoss(blank=len(config['chars']))

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, labels, label_lengths) in enumerate(train_loader):
            # Training steps...
        
        # Validation and metrics collection
        train_acc = calculate_accuracy(model, train_loader, device, config['chars'])
        val_acc = calculate_accuracy(model, val_loader, device, config['chars'])
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

    # Generate plots
    visualizer.plot_loss_curves(train_losses, val_losses)
    visualizer.plot_accuracy(train_accuracies, val_accuracies)
    visualizer.plot_sample_predictions(model, val_loader, device, config['chars'])

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
