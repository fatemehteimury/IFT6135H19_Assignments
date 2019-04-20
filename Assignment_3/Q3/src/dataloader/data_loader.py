import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import dataset
import torch

#############################################################################
#
# Dataset setup
#
#############################################################################

image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5),
                         (.5, .5, .5))
])

image_transform_1 = transforms.Compose([
    transforms.ToTensor(),
])


def get_data_loader(dataset_location, batch_size, normalize):
    trainvalid = torchvision.datasets.SVHN(
        dataset_location, split='train',
        download=True,
        transform=image_transform if normalize else image_transform_1 
    )

    trainset_size = int(len(trainvalid) * 0.9)
    trainset, validset = dataset.random_split(
        trainvalid,
        [trainset_size, len(trainvalid) - trainset_size]
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    validloader = torch.utils.data.DataLoader(
        validset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=True
    )

    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.SVHN(
            dataset_location, split='test',
            download=True,
            transform=image_transform if normalize else image_transform_1 
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=True
    )

    return trainloader, validloader, testloader

