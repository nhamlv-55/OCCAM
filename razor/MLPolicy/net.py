import torch.nn as nn
import torch.nn.functional as F
import torch

def neural_net(type):
    if type=="FeedForward":
        return FeedForward()
    if type=="RNN":
        return RNN()
    if type=="FeedForwardSingleInput":
        return FeedForwardSingleInput()

class FeedForwardSingleInput(nn.Module):
    def __init__(self, features_dim = 35):
        super(FeedForwardSingleInput, self).__init__()
        self.features_dim = features_dim
        self.fc1 = nn.Linear(self.features_dim, self.features_dim/2, bias = True)  # 6*6 from image dimension 
        self.fc2 = nn.Linear(self.features_dim/2, 2, bias = True)
        
    def forward(self, x):
        print("x[-1] in forward:", x[-1])
        h_1 = F.relu(self.fc1(x))
        h_2 = F.relu(self.fc2(h_1))
        output =  F.softmax(h_2, dim = -1)
        return output

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class FeedForward(nn.Module):
    def __init__(self, features_dim = 14, trace_dim = 21):
        super(FeedForward, self).__init__()
        self.features_dim = features_dim
        self.trace_dim = trace_dim
        self.fc1 = nn.Linear(self.features_dim, self.features_dim/2, bias = True)  # 6*6 from image dimension 
        self.fc2 = nn.Linear(self.trace_dim, self.trace_dim/2, bias = True)
        self.fc3 = nn.Linear(self.trace_dim/2+self.features_dim/2, 2, bias = True)
        
    def forward(self, x):
        features = x[:, :14]
        print(features)
        trace = x[:, 14:]
        print(trace)
        h_f = F.relu(self.fc1(features))
        h_t = F.relu(self.fc2(trace))
        print("h_f", h_f, h_f.size())
        print("h_t", h_t, h_t.size())
        concat = torch.cat((h_f, h_t), 1)
        print("concat", concat, concat.size())
        h = self.fc3(concat)
        print("h", h, h.size())
        output =  F.softmax(h, dim = -1)
        print("output", output, output.size())
        return output

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


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
