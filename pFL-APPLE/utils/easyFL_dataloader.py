import numpy as np
import os
import torch
from torch.utils.data import Dataset
import json
from torchvision import datasets, transforms


TRAINDATA = None
TESTDATA = None
TRAINJSON = None
TESTJSON = None


class CustomDataset(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
    

def setData(dataset:str):
    global TRAINDATA
    global TESTDATA
    
    if TRAINDATA is not None:
        return
    
    if dataset.lower() == "mnist":
        TRAINDATA = datasets.MNIST(
            root="../data",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        )
        TESTDATA = datasets.MNIST(
            root="../data",
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        )
        
    elif dataset.lower() == "cifar10":
        TRAINDATA = datasets.CIFAR10(
            root="../data",
            train=True,
            download=False,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
        )

        TESTDATA = datasets.CIFAR10(
            root="../../data",
            train=False,
            download=False,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
        )
        
    elif dataset.lower() == "cifar100":
        TRAINDATA = datasets.CIFAR100(
            root="../data",
            train=True,
            download=False,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
        )

        TESTDATA = datasets.CIFAR100(
            root="../data",
            train=False,
            download=False,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
        )
        
    else:
        raise NotImplementedError
    
    
def set_jsons(folder_path:str):
    global TRAINJSON
    global TESTJSON
    
    if TRAINJSON is None or TESTJSON is None:
        TRAINJSON = json.load(open(os.path.join(folder_path, "train.json")))
        TESTJSON = json.load(open(os.path.join(folder_path, "test.json")))
    
    return len(TRAINDATA.keys())
    
    
def read_client_data(idx):
    # setData(dataset)
    # set_jsons(dataset, folder_path)
    
    global TRAINDATA, TESTDATA, TRAINJSON, TESTJSON
    return CustomDataset(TRAINDATA, TRAINJSON[str(idx)]), CustomDataset(TESTDATA, TESTJSON[str(idx)])