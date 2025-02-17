import os
import torch
from torch import nn, optim
from datetime import datetime
from model import FruitClassifier
from validate import validate_model
from torch.utils.data import DataLoader
from load_data import train_data, val_data
from torch.optim.lr_scheduler import OneCycleLR
from GPUtil import showUtilization as gpu_usage

MODELS_DIR = 'D:/mlfruits/env/models'
os.makedirs(MODELS_DIR, exist_ok=True)

EPOCHS = 75
learning_rate = 0.0003
max_learning_rate = 0.001
weight_decay = 1e-4

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=EPOCHS):
    best_val_acc = 0.0
    patience = 10 
    patience_counter = 0
    
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        val_loss = 0.0
        correct = 0
        total = 0
       
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        results = validate_model(model, val_loader, device, plot_matrices=False)
        val_acc = results['accuracy']

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'best_model.pth'))
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
                    
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        print(f'\nEpoch [{epoch+1}/{epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Best Val Acc: {best_val_acc:.2f}%')
        print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        
        if val_acc > best_val_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'CUDA Version: {torch.version.cuda}')

    model = FruitClassifier()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) 
    
    train_loader = DataLoader(
        train_data, 
        batch_size=8,  
        shuffle=True,
        num_workers=4 if os.name != 'nt' else 0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_data,
        batch_size=16,
        shuffle=False,
        num_workers=4 if os.name != 'nt' else 0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_learning_rate,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )

    final_results = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=EPOCHS)
    
    print("\nFinal Validation Results:")
    results = validate_model(model, val_loader, device, plot_matrices=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'fruit_model_{timestamp}.pth'
    model_path = os.path.join(MODELS_DIR, model_filename)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_path}")

    if torch.cuda.is_available():
        print("\nGPU Usage:")
        gpu_usage()