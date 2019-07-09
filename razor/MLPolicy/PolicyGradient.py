import torch
from net import *
from basePolicy import BasePolicy
import os
import subprocess
import math
from utils import *
import torch.optim as optim
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

class PolicyGradient(BasePolicy):
    def __init__(self, workdir, model_path, network_type, network_hp):
        BasePolicy.__init__(self, workdir, model_path, network_type, network_hp)
        if network_hp is not None:
            self.target_net = network_type(network_hp)
        else:
            self.target_net = network_type()
        self.target_net.load_state_dict(self.net.state_dict())

    def train(self, model_path, no_of_sampling, no_of_iter, from_scratch):
        if from_scratch:
            print("Create a new model")
        else:
            self.net = torch.load(model_path)
        self.optimizer = optim.RMSprop(self.net.parameters())

        for i in range(no_of_iter):
            pass
    def optimize(self):
        pass 
