from sklearn.model_selection import train_test_split
from torchvision.datasets import Omniglot
import torch
import torchvision.transforms as transforms
from setting import args

def get_labels(dataset):
    labels = list()
    for i in range(len(dataset)):
        y = dataset[i][1]
        labels.append(y)
    return labels

# Dataset
transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize(28)
                                ])

train_val_dataset = Omniglot(root=args.root+'/train', background=True, transform=transform, download=True)
test_dataset = Omniglot(root=args.root+'/test', background=False, transform=transform, download=True)

train_val_dataset_labels = get_labels(train_val_dataset)

train_indices, val_indices = train_test_split((range(len(train_val_dataset_labels))), test_size=0.25, shuffle=False)

train_dataset = torch.utils.data.Subset(train_val_dataset, train_indices)
val_dataset = torch.utils.data.Subset(train_val_dataset, val_indices)

train_dataset_labels = get_labels(train_dataset)
val_dataset_labels = get_labels(val_dataset)
test_dataset_labels = get_labels(dataset=test_dataset)
