# file: model.py
import torch
import torch.nn as nn
import os
from pathlib import Path
from torchvision.models import resnet18, ResNet18_Weights


class FruitClassifier(nn.Module):
    def __init__(self, data_dir='D:/mlfruits/env/data'):
        super(FruitClassifier, self).__init__()
        
        # Get number of classes from training directory
        train_dir = Path(data_dir) / 'train'
        val_dir = Path(data_dir) / 'validation'
        
        # Get class folders
        train_classes = set(os.listdir(train_dir))
        val_classes = set(os.listdir(val_dir))
        
        # Verify that training and validation have the same classes
        if train_classes != val_classes:
            raise ValueError(
                f"Training and validation directories must have the same class folders.\n"
                f"Training classes: {sorted(train_classes)}\n"
                f"Validation classes: {sorted(val_classes)}"
            )
        
        num_classes = len(train_classes)
        print(f"Found {num_classes} classes: {sorted(train_classes)}")
        
        # Load ResNet18 with updated weights parameter
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Replace the final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)