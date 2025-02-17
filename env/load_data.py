from torchvision import datasets, transforms
import torchvision.transforms.functional as F

data_dir = 'D:/mlfruits/env/data'

class SquarePad:
    def __call__(self, img):
        w, h = img.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        padding = (hp, vp, hp, vp)
        img = F.pad(img, padding, 0, 'constant')
        
        return F.resize(img, (224, 224))

train_transform = transforms.Compose([
    SquarePad(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(60),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.2, 0.2),
        scale=(0.7, 1.3),
        shear=20
    ),
    transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.2
    ),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    SquarePad(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transform)
val_data = datasets.ImageFolder(data_dir + '/validation', transform=val_transform)

if __name__ == '__main__':
    print(f"Number of training samples: {len(train_data)}")
    print(f"Number of validation samples: {len(val_data)}")
    print(f"Classes: {train_data.classes}")