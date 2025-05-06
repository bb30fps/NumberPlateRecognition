import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.metrics import confusion_matrix

class TrainingVisualizer:
    def __init__(self, output_dir="results"):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        sns.set_style("whitegrid")
        
    def plot_loss_curves(self, train_loss, val_loss):
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title("Training and Validation Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "loss_curves.png"))
        plt.close()

    def plot_accuracy(self, train_acc, val_acc):
        plt.figure(figsize=(10, 6))
        plt.plot(train_acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.title("Training and Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "accuracy_curves.png"))
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, classes):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(os.path.join(self.output_dir, "confusion_matrix.png"))
        plt.close()

    def plot_sample_predictions(self, model, loader, device, chars, num_samples=5):
        model.eval()
        fig, axes = plt.subplots(num_samples, 1, figsize=(10, 15))
        
        with torch.no_grad():
            for i, (images, labels, _) in enumerate(loader):
                if i >= num_samples:
                    break
                
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 2)
                pred_str = self.decode_predictions(preds, chars)[0]
                true_str = self.decode_predictions(labels, chars)[0]
                
                ax = axes[i]
                ax.imshow(images[0].cpu().permute(1, 2, 0))
                ax.set_title(f"Pred: {pred_str} | True: {true_str}")
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "sample_predictions.png"))
        plt.close()

    @staticmethod
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
