import torch
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from load_data import val_data, val_transform
from model import FruitClassifier
from GPUtil import showUtilization as gpu_usage

model_path = 'best_model.pth'

class SquarePad:
    def __call__(self, img):
        # First pad to square
        w, h = img.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        padding = (hp, vp, hp, vp)
        img = F.pad(img, padding, 0, 'constant')
        
        # Then resize to desired size
        return F.resize(img, (224, 224))
    
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.xlabel('predictions')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def validate_model(model, val_loader, device, plot_matrices=False):
    model.eval()
    softmax = torch.nn.Softmax(dim=1)
    correct = 0
    total = 0
    class_correct = {i: 0 for i in range(3)}  # For 3 classes
    class_total = {i: 0 for i in range(3)}
    all_preds = []
    all_labels = []
    misclassified = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images.to(device))
            probabilities = softmax(outputs)
            confidences, predictions = torch.max(probabilities, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            # Collect misclassified images
            for i, (pred, conf, true, img) in enumerate(zip(predictions, confidences, labels, images)):
                if pred != true:
                    misclassified.append({
                        'image': img,
                        'true': val_data.classes[true],
                        'pred': val_data.classes[pred],
                        'conf': conf.item()
                    })
                
                # Per-class accuracy
                class_total[true.item()] += 1
                if pred == true:
                    class_correct[true.item()] += 1
                
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Print overall accuracy
    accuracy = 100 * correct / total
    print(f'\nOverall Validation Accuracy: {accuracy:.2f}%')
    
    # Print per-class accuracy
    print("\nPer-class Accuracy:")
    class_accuracies = {}
    for i in range(3):
        class_acc = 100 * class_correct[i] / class_total[i]
        class_accuracies[val_data.classes[i]] = class_acc
        print(f'{val_data.classes[i]}: {class_acc:.2f}%')

    # Only plot if explicitly requested
    if plot_matrices:
        # Plot confusion matrix
        plot_confusion_matrix(all_labels, all_preds, val_data.classes)
    
    return {
        'accuracy': accuracy,
        'class_accuracies': class_accuracies,
        'confusion_matrix': (all_labels, all_preds),
        'misclassified': misclassified
    }
    

def plot_misclassified(misclassified, max_images=5):
    """Plot misclassified images with their true and predictions labels"""
    if not misclassified:
        print("No misclassified images found!")
        return
    
    # Limit number of images to display
    misclassified = misclassified[:max_images]
        
    fig, axes = plt.subplots(1, len(misclassified), figsize=(15, 3))
    # Handle case where there's only one misclassified image
    if len(misclassified) == 1:
        axes = [axes]
        
    for i, data in enumerate(misclassified):
        # Denormalize image
        img = data['image'].cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f'True: {data["true"]}\nPred: {data["pred"]}\nConf: {data["conf"]*100:.1f}%')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize model and move to device
    model = FruitClassifier()
    model.to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found")
        exit(1)
        
    val_loader = DataLoader(
        val_data, 
        batch_size=16, 
        shuffle=False,
        num_workers=0 if os.name == 'nt' else 4,
        pin_memory=torch.cuda.is_available()
    )
    
    # Run validation
    results = validate_model(model, val_loader, device, plot_matrices=True)

    # Plot misclassified images
    plot_misclassified(results['misclassified'], max_images=8)
    
    # Print GPU usage if available
    if torch.cuda.is_available():
        print("\nGPU Usage:")
        gpu_usage()