import torch
import os

from FLAlgorithms.users.userperavg import UserPerAvg
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import json
from utils.dataloader import CustomDataset
from torchvision import datasets, transforms
from utils.train_smt import test

train_dataset = datasets.MNIST(
    root="./mdata/data",
    download=False,
    train=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
)

test_dataset = datasets.MNIST(
    root="./mdata/data",
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
)


def load_total_dataset(folder_path):
    return json.load(open(folder_path + "/mnist_sparse.json","r")), json.load(open(folder_path + "/mnist_sparse_test.json","r"))


def load_user_dataset(id, train_jsons, test_jsons):
    return id, CustomDataset(train_dataset, train_jsons[str(id)]), CustomDataset(train_dataset, test_jsons[str(id)])

# Implementation for per-FedAvg Server

class PerAvg(Server):
    def __init__(self,device, dataset,algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users,times, json_path=None):
        super().__init__(device, dataset,algorithm, model[0], batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times)

        # Initialize data for all  users
        train_jsons, test_jsons = load_total_dataset(folder_path="./mdata/dataset_idx/mnist/sparse/100client")
        total_users = len(train_jsons.keys())
        
        for i in range(total_users):
            id, train, test = load_user_dataset(i, train_jsons, test_jsons)
            user = UserPerAvg(device, id, train, test, model, batch_size, learning_rate, beta, lamda, local_epochs, optimizer ,total_users , num_users)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating Local Per-Avg.")

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
            print("Evaluate global model with one step update")
            print("")
            self.evaluate_one_step()

            # choose several users to send back upated model to server
            self.selected_users = self.select_users(glob_iter,self.num_users)
            for user in self.selected_users:
                user.train(self.local_epochs) #* user.train_samples
                
            self.aggregate_parameters()

        # self.save_results()
        # self.save_model()
        
            cfmtx = test(self.model, test_dataset)
            global_cfmtx_record.append(cfmtx)
        
        return global_cfmtx_record
