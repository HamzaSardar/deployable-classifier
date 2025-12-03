import torch
import torchvision 

def get_dataloaders(
    batch_size: int=32,
    n_workers: int=2
):
    """
    Loads and returns train and test dataloader for CIFAR10 using torchvision.datasets.
    """
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=n_workers)

    test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=n_workers)

    return train_loader, test_loader
    
