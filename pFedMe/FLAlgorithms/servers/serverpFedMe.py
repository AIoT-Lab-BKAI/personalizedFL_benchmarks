import torch
import os

from FLAlgorithms.users.userpFedMe import UserpFedMe
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np
import json
from utils.dataloader import CustomDataset
from torchvision import datasets, transforms
from utils.train_smt import test

# Implementation for pFedMe Server
train_dataset = datasets.CIFAR10(
    root="./mdata/data",
    download=True,
    train=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
)

test_dataset = datasets.CIFAR10(
    root="./mdata/data",
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
)


def load_total_dataset(folder_path):
    return json.load(open(folder_path + "/cifar10_sparse.json","r")), json.load(open(folder_path + "/cifar10_sparse_test.json","r"))


def load_user_dataset(id, train_jsons, test_jsons):
    return id, CustomDataset(train_dataset, train_jsons[str(id)]), CustomDataset(train_dataset, test_jsons[str(id)])


class pFedMe(Server):
    def __init__(self, device,  dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, K, personal_learning_rate, times, json_path=None):
        super().__init__(device, dataset,algorithm, model[0], batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times)

        # Initialize data for all  users
        train_jsons, test_jsons = load_total_dataset(folder_path="./mdata/dataset_idx/cifar10/sparse/100client")
        total_users = len(train_jsons.keys())
        self.K = K
        self.personal_learning_rate = personal_learning_rate
        for i in range(total_users):
            # id, train , test = read_user_data(i, data, dataset)
            id, train, test = load_user_dataset(i, train_jsons, test_jsons)
            
            user = UserpFedMe(device, id, train, test, model, batch_size, learning_rate, beta, lamda, local_epochs, optimizer, K, personal_learning_rate)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating pFedMe server.")

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def train(self):
        loss = []
        global_cfmtx_record = []
        
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            # send all parameter for users 
            self.send_parameters()

            # Evaluate gloal model on user for each interation
            print("Evaluate global model")
            print("")
            self.evaluate()

            # do update for all users not only selected users
            for user in self.users:
                user.train(self.local_epochs) #* user.train_samples
            
            # choose several users to send back upated model to server
            self.selected_users = self.select_users(glob_iter, self.num_users)

            self.evaluate_personalized_model()
            self.persionalized_aggregate_parameters()

            cfmtx = test(self.model, test_dataset)
            global_cfmtx_record.append(cfmtx)
        
        return global_cfmtx_record

    
  
