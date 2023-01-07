import torch
import torch.nn as nn
import torch.nn.functional as F


class mnistNet(nn.Module):
    """ CNN
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encoder(self, x):
        x = x.view((x.shape[0],28,28))
        x = x.unsqueeze(1)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        return x

    def decoder(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class cifar10Net(nn.Module):
    """ CNN
    """
    def __init__(self):
        super().__init__()
        self.encoder_block = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.decoder_block = nn.Sequential(
            nn.Linear(1600, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, 10),
        )
    
    def encoder(self, x):
        x = self.encoder_block(x)
        x = x.flatten(1)
        return x
    
    def decoder(self, x):
        x = self.decoder_block(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    
def conv_block(in_channel, out_channel, pool=False):
    layers = [
        nn.Conv2d(in_channel,out_channel,kernel_size=3,padding=1,stride=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
        ]

    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)
        

class cifar100Net(nn.Module):
    """ Resnet 9
    """
    def __init__(self, in_channel=3, num_classes=100):
        super().__init__()

        self.conv1 = conv_block(in_channel,32)
        self.conv2 = conv_block(32,64,pool=True)
        self.res1 = nn.Sequential(conv_block(64,64),conv_block(64,64))

        self.conv3 = conv_block(64,128,pool=True)
        self.conv4 = conv_block(128,256,pool=True)
        self.res2 = nn.Sequential(conv_block(256,256),conv_block(256,256))

        self.pool = nn.MaxPool2d(4)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self,x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out
    
    def encoder(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.pool(out)
        out = self.dropout(out.view(x.shape[0], -1))
        return out
    
    def decoder(self, out):
        out = self.classifier(out)
        return out
        
