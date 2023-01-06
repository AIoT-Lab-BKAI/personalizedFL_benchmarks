import numpy as np
import os
import torch
from torch.utils.data import Dataset
import json
from torchvision import datasets, transforms

# IMAGE_SIZE = 28
# IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
# NUM_CHANNELS = 1

# IMAGE_SIZE_CIFAR = 32
# NUM_CHANNELS_CIFAR = 3

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
    

def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    ran_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(ran_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)


def get_random_batch_sample(data_x, data_y, batch_size):
    num_parts = len(data_x)//batch_size + 1
    if(len(data_x) > batch_size):
        batch_idx = np.random.choice(list(range(num_parts + 1)))
        sample_index = batch_idx*batch_size
        if(sample_index + batch_size > len(data_x)):
            return (data_x[sample_index:], data_y[sample_index:])
        else:
            return (data_x[sample_index: sample_index+batch_size], data_y[sample_index: sample_index+batch_size])
    else:
        return (data_x, data_y)


def get_batch_sample(data, batch_size):
    data_x = data['x']
    data_y = data['y']

    # np.random.seed(100)
    ran_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(ran_state)
    np.random.shuffle(data_y)

    batched_x = data_x[0:batch_size]
    batched_y = data_y[0:batch_size]
    return (batched_x, batched_y)


# def read_data(dataset, idx, is_train=True):
#     if is_train:
#         train_data_dir = os.path.join('../dataset', dataset, 'train/')

#         train_file = train_data_dir + 'train' + str(idx) + '_.npz'
#         with open(train_file, 'rb') as f:
#             train_data = np.load(f, allow_pickle=True)['data'].tolist()

#         return train_data

#     else:
#         test_data_dir = os.path.join('../dataset', dataset, 'test/')

#         test_file = test_data_dir + 'test' + str(idx) + '_.npz'
#         with open(test_file, 'rb') as f:
#             test_data = np.load(f, allow_pickle=True)['data'].tolist()

#         return test_data


def setData(dataset:str):
    global TRAINDATA
    global TESTDATA
    
    if TRAINDATA is not None:
        return
    
    if dataset.lower() == "mnist":
        TRAINDATA = datasets.MNIST(
            root="../../data",
            train=True,
            download=False,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        )
        TESTDATA = datasets.MNIST(
            root="../../data",
            train=False,
            download=False,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        )
        
    elif dataset.lower() == "cifar10":
        TRAINDATA = datasets.CIFAR10(
            root="../../data",
            train=True,
            download=False,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
        )

        TESTDATA = datasets.CIFAR10(
            root="../../../data",
            train=False,
            download=False,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
        )
        
    elif dataset.lower() == "cifar100":
        TRAINDATA = datasets.CIFAR100(
            root="../../data",
            train=True,
            download=False,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
        )

        TESTDATA = datasets.CIFAR100(
            root="../../data",
            train=False,
            download=False,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
        )
        
    else:
        raise NotImplementedError
    
    
def set_jsons(dataset:str, folder_path:str):
    global TRAINJSON
    global TESTJSON
    
    if TRAINJSON is None or TESTJSON is None:
        TRAINJSON = json.load(open(os.path.join(folder_path, "train.json")))
        TESTJSON = json.load(open(os.path.join(folder_path, "test.json")))
    return
    
    
def read_client_data(dataset: str, idx, is_train=True, folder_path=None):
    setData(dataset)
    set_jsons(dataset, folder_path)
    
    global TRAINDATA, TESTDATA, TRAINJSON, TESTJSON
    if is_train:
        return CustomDataset(TRAINDATA, TRAINJSON[str(idx)])
    else:
        return CustomDataset(TESTDATA, TESTJSON[str(idx)])


# def read_client_data_text(dataset, idx, is_train=True):
#     if is_train:
#         train_data = read_data(dataset, idx, is_train)
#         X_train, X_train_lens = list(zip(*train_data['x']))
#         y_train = train_data['y']

#         X_train = torch.Tensor(X_train).type(torch.int64)
#         X_train_lens = torch.Tensor(X_train_lens).type(torch.int64)
#         y_train = torch.Tensor(train_data['y']).type(torch.int64)

#         train_data = [((x, lens), y) for x, lens, y in zip(X_train, X_train_lens, y_train)]
#         return train_data
#     else:
#         test_data = read_data(dataset, idx, is_train)
#         X_test, X_test_lens = list(zip(*test_data['x']))
#         y_test = test_data['y']

#         X_test = torch.Tensor(X_test).type(torch.int64)
#         X_test_lens = torch.Tensor(X_test_lens).type(torch.int64)
#         y_test = torch.Tensor(test_data['y']).type(torch.int64)

#         test_data = [((x, lens), y) for x, lens, y in zip(X_test, X_test_lens, y_test)]
#         return test_data
