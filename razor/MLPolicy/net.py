from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
import torch

DEBUG = False

class Net(nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.features_len = dataset.features_len
        self.mean = torch.tensor(dataset.mean, dtype = torch.float).view(1, -1)
        self.std  = torch.tensor(dataset.std, dtype = torch.float).view(1, -1)
        print("mean when init network:", self.mean)
        print("std when init network:", self.std)
class FeedForwardSingleInput(Net):
    def __init__(self, dataset):
        Net.__init__(self, dataset)
        print(self.features_len)
        self.fc_f1 = nn.Linear(self.features_len, self.features_len/2, bias = True)  
        self.fc_f2 = nn.Linear(self.features_len/2, self.features_len/4, bias = True)
        self.fc_f3 = nn.Linear(self.features_len/4, 2, bias = True)
    def forward(self, x): 
        print(x, x.size())
        x -= self.mean
        x /= self.std
        #print("x after normalization:", x)
        #features = x.to(torch.float)
        #print("x after normalization:", features)
        h_f = F.relu(self.fc_f1(x))
        h_f2 = F.relu(self.fc_f2(h_f))
        output = self.fc_f3(h_f2)
        return output

class RNN(nn.Module):
    def __init__(self, input_dim = 15, hidden_dim = 10, type = "LSTM"):
        super(RNN, self).__init__()
        self.INPUT_DIM = input_dim
        if type=="LSTM":
            self.rnn = nn.LSTM(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 10, bias = True)
        self.fc2 = nn.Linear(10, 1, bias = True)

    def forward(self, inputs):
        _, (last_h, last_c) = self.rnn(inputs)
        fc1 = F.relu(self.fc1(last_h))
        prediction = self.fc2(fc1)
        return prediction
