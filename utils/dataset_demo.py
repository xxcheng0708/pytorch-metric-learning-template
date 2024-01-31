import torchvision
from torch.utils.data import DataLoader

train_transforms_list = [
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
val_transforms_list = [
    torchvision.transforms.Resize((224, 224)),
    # torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
train_transforms = torchvision.transforms.Compose(train_transforms_list)
val_transforms = torchvision.transforms.Compose(val_transforms_list)

train_dataset = torchvision.datasets.CIFAR10(root="./datasets", train=True, transform=train_transforms, download=True)
val_dataset = torchvision.datasets.CIFAR10(root="./datasets", train=False, transform=val_transforms, download=True)
classes = train_dataset.classes

train_data_loader = DataLoader(train_dataset, batch_size=16, drop_last=True, shuffle=True,
                               num_workers=4)
val_data_loader = DataLoader(train_dataset, batch_size=16, drop_last=False, shuffle=False,
                              num_workers=4)

for x, y in train_data_loader:
    print(x)
    break