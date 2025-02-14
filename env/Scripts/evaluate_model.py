import torch
from model import FruitClassifier
from load_data import val_data
from torch.utils.data import DataLoader
from validate import validate_model
from GPUtil import showUtilization as gpu_usage
import os

def evaluate_saved_model(model_path):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize model
    model = FruitClassifier()
    model.to(device)
    
    # Load saved model
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found")
        return
        
    # Create validation loader
    val_loader = DataLoader(
        val_data,
        batch_size=8,
        shuffle=False,
        num_workers=0 if os.name == 'nt' else 4,
        pin_memory=torch.cuda.is_available()
    )
    
    # Run validation with plots
    model.eval()
    results = validate_model(model, val_loader, device, plot_matrices=True)
    
    if torch.cuda.is_available():
        print("\nGPU Usage:")
        print(gpu_usage())

if __name__ == "__main__":

    model_path = 'best_model.pth' 
    evaluate_saved_model(model_path)