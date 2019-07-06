import torch.nn as nn
import torch.nn.functional as F
import torch


DEBUG = False

def neural_net(type):
    if type=="FeedForward":
        return FeedForward()
    if type=="RNN":
        return RNN()
    if type=="FeedForwardSingleInput":
        return FeedForwardSingleInput()

class FeedForwardSingleInput(nn.Module):
    def __init__(self, features_dim = 14, trace_dim = 21*2, state = None):
        print(features_dim)
        super(FeedForwardSingleInput, self).__init__()
        self.features_dim = features_dim
        self.trace_dim = trace_dim
        if state is None:
            self.state = torch.tensor([0]*features_dim*(trace_dim/2), dtype = torch.float).view(-1, features_dim*(trace_dim/2))
        else:
            print("total state:")
            if DEBUG:
                for s in state:
                    print(s)
            self.state = torch.tensor(state).view(-1, len(state))
        print(self.state.size())
        self.state_dim = self.state.size()[-1]
        #fc for state
        #self.fc_s1 = nn.Linear(self.state_dim, self.state_dim/2, bias = True)
        #self.fc_s2 = nn.Linear(self.state_dim/2, self.state_dim/4, bias = True)
        #fc for features
        self.fc_f1 = nn.Linear(self.features_dim, self.features_dim/2, bias = True)  # 6*6 from image dimension 
        self.fc_f2 = nn.Linear(self.features_dim/2, self.features_dim/4, bias = True)
        self.fc_f3 = nn.Linear(self.features_dim/4, 2, bias = True)
        #fc for trace
        #self.fc_t1 = nn.Linear(self.trace_dim, self.trace_dim/2, bias = True)

        #output
        #self.fc_o = nn.Linear(self.state_dim/4 + self.features_dim/4 + self.trace_dim/2, 2, bias = True)
        
        
    def forward(self, x): 
        #print(x, x.size())
        features = x[:, :self.features_dim]
        trace = x[:,self.features_dim:]
        batch_size = x.size()[0]
        #print("batch_size", batch_size)
        state_tiled = self.state.repeat((batch_size, 1))
        #print(self.state.size())
        #trace = trace.view(-1, 21, 2)
        #print("trace", trace)
        #print("x[-1] in forward:", x[-1])
        h_f = F.relu(self.fc_f1(features))
        h_f2 = F.relu(self.fc_f2(h_f))

        #h_t = F.relu(self.fc_t1(trace))

        #h_s = F.relu(self.fc_s1(state_tiled))
        #h_s2 = F.relu(self.fc_s2(h_s))
        #concat
        #print(h_f2.size(), h_t.size(), h_s2.size())
        #concat = torch.cat((h_f2, h_t, h_s2), 1)
        
        #output =  F.softmax(self.fc_o(concat), dim = -1)
        output = self.fc_f3(h_f2)
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
