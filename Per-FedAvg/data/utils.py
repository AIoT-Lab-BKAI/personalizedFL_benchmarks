import pickle
import os
from torch.utils.data import random_split, DataLoader
from dataset import MNISTDataset, CIFARDataset
from path import Path
from torchvision import datasets, transforms

DATASET_DICT = {
    "mnist": MNISTDataset,
    "cifar": CIFARDataset,
}
CURRENT_DIR = Path(__file__).parent.abspath()

# test_dataset = datasets.MNIST(
#     root="./mdata/data",
#     train=False,
#     download=True,
#     transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
# )

def get_dataloader(dataset: str, client_id: int, batch_size=20, valset_ratio=0.1):
    # pickles_dir = CURRENT_DIR / dataset / "pickles"
    # if os.path.isdir(pickles_dir) is False:
    #     raise RuntimeError("Please preprocess and create pickles first.")

    # with open(pickles_dir / str(client_id) + ".pkl", "rb") as f:
    #     client_dataset: DATASET_DICT[dataset] = pickle.load(f)

    # val_num_samples = int(valset_ratio * len(client_dataset))
    # train_num_samples = len(client_dataset) - val_num_samples

    # trainset, valset = random_split(
    #     client_dataset, [train_num_samples, val_num_samples]
    # )
    # trainloader = DataLoader(trainset, batch_size, drop_last=True)
    # valloader = DataLoader(valset, batch_size)
    
    data = None
    if dataset == "mnist":
        data = datasets.MNIST(
            root="./torchdata",
            download=True,
            train=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        )
    elif dataset == "cifar10":
        data = datasets.CIFAR10(
            root="./torchdata", 
            train=True, 
            download=True, 
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        )
    else:
        raise Exception("Not support" + dataset)

    return trainloader, valloader


def get_client_id_indices(dataset):
    dataset_pickles_path = CURRENT_DIR / dataset / "pickles"
    with open(dataset_pickles_path / "seperation.pkl", "rb") as f:
        seperation = pickle.load(f)
    return (seperation["train"], seperation["test"], seperation["total"])

