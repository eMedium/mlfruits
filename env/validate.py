import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from GPUtil import showUtilization as gpu_usage
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from model import FruitClassifier
from load_data import val_data


MODELS_DIR = 'D:/mlfruits/env/models'
model_path = os.path.join(MODELS_DIR, 'best_model.pth')

class SquarePad:
    def __call__(self, img):
        w, h = img.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        padding = (hp, vp, hp, vp)
        img = F.pad(img, padding, 0, 'constant')
        
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
    num_classes = len(val_data.classes)
    class_correct = {i: 0 for i in range(num_classes)}
    class_total = {i: 0 for i in range(num_classes)}
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

            for i, (pred, conf, true, img) in enumerate(zip(predictions, confidences, labels, images)):
                if pred != true:
                    misclassified.append({
                        'image': img,
                        'true': val_data.classes[true],
                        'pred': val_data.classes[pred],
                        'conf': conf.item()
                    })
                
                class_total[true.item()] += 1
                if pred == true:
                    class_correct[true.item()] += 1
                
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f'\nOverall Validation Accuracy: {accuracy:.2f}%')
    
    print("\nPer-class Accuracy:")
    class_accuracies = {}
    for i in range(num_classes):
        class_acc = 100 * class_correct[i] / class_total[i]
        class_accuracies[val_data.classes[i]] = class_acc
        print(f'{val_data.classes[i]}: {class_acc:.2f}%')

    if plot_matrices:
        plot_confusion_matrix(all_labels, all_preds, val_data.classes)
    
    return {
        'accuracy': accuracy,
        'class_accuracies': class_accuracies,
        'confusion_matrix': (all_labels, all_preds),
        'misclassified': misclassified
    }
    

def plot_misclassified(misclassified, max_images=5):
    if not misclassified:
        print("No misclassified images found!")
        return
    
    misclassified = misclassified[:max_images]
        
    fig, axes = plt.subplots(1, len(misclassified), figsize=(15, 3))
    if len(misclassified) == 1:
        axes = [axes]
        
    for i, data in enumerate(misclassified):
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

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
    
    results = validate_model(model, val_loader, device, plot_matrices=True)

    plot_misclassified(results['misclassified'], max_images=8)
    
    if torch.cuda.is_available():
        print("\nGPU Usage:")
        gpu_usage()