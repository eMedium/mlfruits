import os
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder

def analyze_dataset(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'validation')

    if not os.path.exists(train_dir):
        raise ValueError(f"Training directory not found: {train_dir}")
    if not os.path.exists(val_dir):
        raise ValueError(f"Validation directory not found: {val_dir}")
    
    train_data = ImageFolder(train_dir)
    val_data = ImageFolder(val_dir)
    
    train_counts = {train_data.classes[i]: 0 for i in range(len(train_data.classes))}
    val_counts = {val_data.classes[i]: 0 for i in range(len(val_data.classes))}
    
    for _, label in train_data:
        train_counts[train_data.classes[label]] += 1
    for _, label in val_data:
        val_counts[val_data.classes[label]] += 1
    
    plt.figure(figsize=(10, 5))
    x = range(len(train_counts))
    width = 0.35
    
    train_bars = plt.bar([i - width/2 for i in x], train_counts.values(), width, label='Training')
    val_bars = plt.bar([i + width/2 for i in x], val_counts.values(), width, label='Validation')
    
    plt.bar_label(train_bars, label_type='center')
    plt.bar_label(val_bars, label_type='center')

    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.title('Dataset Distribution')
    plt.xticks(x, train_counts.keys())
    plt.legend()
    plt.show()
    
    print("\nDataset Statistics:")
    print("\nTraining Set:")
    for cls, count in train_counts.items():
        print(f"{cls}: {count} images")
    
    print("\nValidation Set:")
    for cls, count in val_counts.items():
        print(f"{cls}: {count} images")

if __name__ == "__main__":
    analyze_dataset("D:/mlfruits/env/data")