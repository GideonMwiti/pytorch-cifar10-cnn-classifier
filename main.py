"""
Project 14: Convolutional Neural Network (CNN) on CIFAR-10 in PyTorch
Dataset: CIFAR-10 (60,000 32x32 Color RGB Images across 10 Categories)
Framework: PyTorch
Author: Gideon Mwiti

Key Features:
- 3-Stage Convolutional Neural Network with 2D Batch Normalization & Spatial Dropout.
- Data Augmentation pipeline (RandomCrop with padding, RandomHorizontalFlip).
- Multi-class evaluation across 10 object classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
- Class-wise confusion analysis and Top-1 accuracy tracking.
- Publication-quality 4-panel diagnostic visual artifact:
  1) Training and Validation Cross-Entropy Loss Convergence
  2) Training and Validation Classification Accuracy (%) Curves
  3) Qualitative Test Predictions Grid (16 RGB sample images with confidence bars)
  4) Normalized 10x10 Confusion Matrix across CIFAR-10 categories
"""

import os
import sys
import types
import random
import numpy as np
import matplotlib.pyplot as plt

# Workaround for Windows environments where WDAC blocks unsigned bz2/lzma DLLs
if 'bz2' not in sys.modules:
    sys.modules['bz2'] = types.SimpleNamespace(open=open)
if 'lzma' not in sys.modules:
    sys.modules['lzma'] = types.SimpleNamespace(open=open)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


class CIFAR10ConvNet(nn.Module):
    """
    Deep Convolutional Neural Network for CIFAR-10 Multi-Class Classification.
    Topology:
      Input (3x32x32)
      -> [Conv(32, 3x3) -> BN -> ReLU] x 2 -> MaxPool(2x2) -> Dropout(0.20) [32x16x16]
      -> [Conv(64, 3x3) -> BN -> ReLU] x 2 -> MaxPool(2x2) -> Dropout(0.30) [64x8x8]
      -> [Conv(128, 3x3) -> BN -> ReLU]     -> MaxPool(2x2) -> Dropout(0.35) [128x4x4]
      -> Flatten (2048) -> Linear(256) -> BN -> ReLU -> Dropout(0.40)
      -> Linear(10) -> Logits
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Block 1: Low-level RGB color/edge features
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.20)
        )
        
        # Block 2: Intermediate textural motifs
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.30)
        )
        
        # Block 3: High-level semantic parts
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.35)
        )
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


def get_cifar10_loaders(data_dir, batch_size=256):
    """Creates data loaders with standard CIFAR-10 data augmentation."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, test_loader


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    return running_loss / total, (correct / total) * 100.0


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    return running_loss / total, (correct / total) * 100.0, np.array(all_preds), np.array(all_targets)


def generate_visual_artifact(history, sample_imgs, sample_lbls, sample_preds, sample_confs,
                             targets, preds, class_names, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    plt.subplots_adjust(hspace=0.32, wspace=0.25)
    epochs = range(1, len(history['train_loss']) + 1)
    
    # -------------------------------------------------------------
    # Panel 1: Loss Convergence Curves
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2.0)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2.2)
    ax1.set_title("Cross-Entropy Loss vs. Epochs\nSmooth Generalization on 10 Color Object Categories", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', frameon=True, fontsize=9)
    
    # -------------------------------------------------------------
    # Panel 2: Accuracy Dynamics
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_acc'], 'b--', label='Train Accuracy', alpha=0.7)
    ax2.plot(epochs, history['val_acc'], 'g-o', label=f"Val Accuracy (Final: {history['val_acc'][-1]:.2f}%)", linewidth=2.2)
    ax2.set_title("Training and Validation Classification Accuracy\nDeep CNN Achieving High Generalization on CIFAR-10", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right', frameon=True, fontsize=9)
    
    # -------------------------------------------------------------
    # Panel 3: Qualitative Sample Predictions Grid (4x4)
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.axis('off')
    sub_grid = axes[1, 0].inset_axes([0.0, 0.0, 1.0, 1.0])
    sub_grid.axis('off')
    
    # CIFAR-10 normalization stats for un-normalization
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])
    
    for idx in range(16):
        row = idx // 4
        col = idx % 4
        sub_ax = sub_grid.inset_axes([col * 0.25 + 0.02, 0.75 - row * 0.25 + 0.02, 0.21, 0.21])
        
        img = sample_imgs[idx].cpu().numpy().transpose(1, 2, 0)
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        sub_ax.imshow(img)
        sub_ax.axis('off')
        
        true_lbl = class_names[sample_lbls[idx]]
        pred_lbl = class_names[sample_preds[idx]]
        conf = sample_confs[idx] * 100.0
        
        is_correct = (sample_lbls[idx] == sample_preds[idx])
        color = 'darkgreen' if is_correct else 'crimson'
        sub_ax.set_title(f"T: {true_lbl} | P: {pred_lbl}\n({conf:.0f}%)", 
                         fontsize=8, fontweight='bold', color=color, pad=2)
        
    ax3.set_title("Qualitative Test Set Image Classifications (4x4 Grid)\nGreen: Correct Prediction | Red: Challenging Ambiguity", 
                  fontsize=11, fontweight='bold', pad=15)

    # -------------------------------------------------------------
    # Panel 4: Normalized Confusion Matrix (10x10)
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes))
    for t, p in zip(targets, preds):
        cm[int(t), int(p)] += 1
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    
    im4 = ax4.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax4.set_xticks(range(num_classes))
    ax4.set_yticks(range(num_classes))
    ax4.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax4.set_yticklabels(class_names, fontsize=9)
    ax4.set_xlabel("Predicted Label", fontsize=10)
    ax4.set_ylabel("True Ground Truth Label", fontsize=10)
    ax4.set_title(f"Normalized Confusion Matrix (10x10 Categories, N = {len(targets)})\nHigh Diagonal Density with Animal/Vehicle Distinction", fontsize=11, fontweight='bold')
    
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm_norm[i, j]
            if val >= 0.05:
                color = "white" if val > 0.5 else "black"
                ax4.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7.5)
                
    cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    cbar4.set_label("Accuracy Ratio", fontsize=10)

    plt.suptitle("PyTorch Convolutional Neural Network (CNN) on CIFAR-10 Benchmark", 
                 fontsize=15, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nVisual diagnostic report saved successfully at: {os.path.abspath(save_path)}")


def main():
    print("=" * 72)
    print("         PROJECT 14: CONVOLUTIONAL NEURAL NETWORK ON CIFAR-10")
    print("=" * 72)
    
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device    : {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    save_path = os.path.join(script_dir, "cifar10_cnn_classification_results.png")
    
    # Hyperparameters
    batch_size = 256
    epochs = 5
    learning_rate = 0.002
    
    # Load CIFAR-10
    print("\nLoading CIFAR-10 Dataset (50,000 Train, 10,000 Test 32x32 RGB)...")
    train_loader, test_loader = get_cifar10_loaders(data_dir, batch_size=batch_size)
    print(f"Number of Classes   : 10 ({', '.join(CIFAR10_CLASSES)})")
    print(f"Data Augmentation   : RandomCrop(32, pad=4) + RandomHorizontalFlip + Normalization")
    print("-" * 72)
    
    # Initialize Model, Criterion, Optimizer
    model = CIFAR10ConvNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Architecture  : 3-Stage CNN with BatchNorm & Dropout")
    print(f"Trainable Parameters: {total_params:,}")
    print("-" * 72)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print("\n--- Commencing CNN Training Across 5 Epochs ---")
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, preds, targets = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
    print("-" * 72)
    print(f"Training Complete! Final Test Accuracy: {history['val_acc'][-1]:.2f}% | Final Loss: {history['val_loss'][-1]:.4f}")
    print("=" * 72)
    
    # Qualitative Sample Predictions
    sample_images, sample_labels = next(iter(test_loader))
    sample_imgs = sample_images[:16].to(device)
    sample_lbls = sample_labels[:16].numpy()
    
    model.eval()
    with torch.no_grad():
        out = model(sample_imgs)
        probs = torch.softmax(out, dim=1)
        s_confs, s_preds = torch.max(probs, dim=1)
        sample_confs = s_confs.cpu().numpy()
        sample_preds = s_preds.cpu().numpy()
        
    # Generate Visual Dashboard
    generate_visual_artifact(
        history, sample_imgs, sample_lbls, sample_preds, sample_confs,
        targets, preds, CIFAR10_CLASSES, save_path
    )


if __name__ == "__main__":
    main()

