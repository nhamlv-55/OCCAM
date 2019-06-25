import torch.nn as nn
import torch.nn.functional as F
import torch

def neural_net(type):
    if type=="FeedForward":
        return FeedForward()
    if type=="RNN":
        return RNN()


class FeedForward(nn.Module):
    def __init__(self, input_dim = 14):
        super(Net, self).__init__()
        self.INPUT_DIM = input_dim
        self.fc1 = nn.Linear(self.INPUT_DIM, self.INPUT_DIM//2, bias = True)  # 6*6 from image dimension 
        self.fc1.to(torch.double)
        self.fc2 = nn.Linear(self.INPUT_DIM//2, 2, bias = True)
        self.fc2.to(torch.double)
        
    def forward(self, x):
        x = x.view(-1, self.INPUT_DIM)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class RNN(nn.Module):
    def __init__(self, input_dim = 14, hidden_dim = 10, type = "LSTM"):
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
