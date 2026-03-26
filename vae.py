import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pyro.contrib.examples.util
from pyro.contrib.examples.util import MNIST

class Decoder(nn.Module):

    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, 784)

        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        hidden = self.softplus(self.fc1(z))
        loc_img = self.sigmoid(self.fc21(hidden))
        return loc_img

class Encoder(nn.Module):
    
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)

        self.softplus = nn.Softplus()

    def forward(self, x):
        x = x.reshape(-1, 784)
        hidden = self.softplus(self.fc1(x))
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return z_loc, z_scale

class VAE(nn.Module):
    def __init__(self, z_dim=50, hidden_dim=400 ,*args, **kwargs):
        super().__init__(*args, **kwargs)



def dataloader_setup(batch_size = 128):
    root = './data'
    download = True
    trans = transforms.ToTensor()

    training_set = MNIST(root=root, train=True, transform=trans, download=download)
    testing_set = MNIST(root=root, train=False, transform=trans, download=download)

    train_loader = DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=testing_set, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


