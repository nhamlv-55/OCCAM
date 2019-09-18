from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
import torch

DEBUG = False
#DEBUG = True

torch.set_printoptions(sci_mode = False)

class Net(nn.Module):
    def __init__(self, metadata):
        super(Net, self).__init__()
        self.features_len = metadata["features_len"]
        self.mean = torch.tensor(metadata["mean"], dtype = torch.float).view(1, -1)
        self.std  = torch.tensor(metadata["std"], dtype = torch.float).view(1, -1)

        print("mean when init network:", self.mean)
        print("std when init network:", self.std)

class TinyFFNSoftmax(Net):
    #Use tanh for easier training (only with tiny net)
    def __init__(self, metadata):
        Net.__init__(self, metadata)
        #print("features len:", self.features_len)
        self.fc_f1 = nn.Linear(self.features_len, self.features_len, bias = True)  
        self.fc_f2 = nn.Linear(self.features_len, 2, bias = True)
    def forward(self, x): 
        if DEBUG: print(x, x.size())
        features = x - self.mean
        #if DEBUG: print("x after subtract mean:", features)
        features /= self.std
        #print("x after normalization:", x)
        features = x.to(torch.float)
        #if DEBUG: print("x after normalization:", features)
        h_f = F.relu(self.fc_f1(x))
        #if DEBUG: print("h_f:",h_f)
        h_f2 = F.relu(self.fc_f2(x))
        if DEBUG: print("h_f2:", h_f2)
        output = F.softmax(h_f2, dim = -1)
        return output

class FeedForwardSingleInput(Net):
    def __init__(self, metadata):
        Net.__init__(self, metadata)
        print(self.features_len)
        self.fc_f1 = nn.Linear(self.features_len, self.features_len, bias = True)  
        self.fc_f2 = nn.Linear(self.features_len, self.features_len/2, bias = True)
        self.fc_f3 = nn.Linear(self.features_len/2, self.features_len/4, bias = True)
        self.fc_f4 = nn.Linear(self.features_len/4, 2, bias = True)
    def forward(self, x): 
        if DEBUG: print(x, x.size())
        features = x - self.mean
        if DEBUG: print("x after subtract mean:", features)
        features /= self.std
        #print("x after normalization:", x)
        #features = x.to(torch.float)
        if DEBUG: print("x after normalization:", features)
        h_f = F.relu(self.fc_f1(features))
        h_f2 = F.relu(self.fc_f2(h_f))
        h_f3 = F.relu(self.fc_f3(h_f2))
        output = self.fc_f4(h_f3)
        print(output)
        return output

class FeedForwardSingleInputSoftmax(Net):
    def __init__(self, metadata):
        Net.__init__(self, metadata)
        print(self.features_len)
        self.fc_f1 = nn.Linear(self.features_len, self.features_len )  
        self.fc_f2 = nn.Linear(self.features_len, self.features_len/2)
        self.fc_f3 = nn.Linear(self.features_len/2, self.features_len/4)
        self.fc_f4 = nn.Linear(self.features_len/4, 2)
    def forward(self, x): 
        if DEBUG: print(x, x.size())
        features = x- self.mean.view(-1)
        if DEBUG: print("x after subtract mean:", features)
        features /= self.std.view(-1)
        #print("x after normalization:", x)
        #features = x.to(torch.float)
        if DEBUG: print("x after normalization:", features)
        h_f = F.relu(self.fc_f1(features))
        h_f2 = F.relu(self.fc_f2(h_f))
        h_f3 = F.relu(self.fc_f3(h_f2))
        output = self.fc_f4(h_f3)
        output = F.softmax(output, dim = -1)
        if DEBUG: print("output:", output)
        return output

def create_emb_layer(emb_matrix, non_trainable = False):
    num_emb, emb_size = emb_matrix.shape
    print num_emb, emb_size
    emb_layer = nn.Embedding(num_emb, emb_size)
    emb_layer.weight = nn.Parameter(torch.from_numpy(emb_matrix))
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer, num_emb, emb_size

class UberNet(nn.Module):
    def __init__(self, weights_matrix, hidden_size, num_layers):
        super(self).__init__()
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru_caller = nn.GRU(embedding_dim, hidden_size_caller, num_layers_caller, batch_first = True)
        self.gru_callee = nn.GRU(embedding_dim, hidden_size_callee, num_layers_callee, batch_first = True)
        self.gru_args   = nn.GRU(embedding_dim_args, hidden_size_args, num_layers_args, batch_first = True)

        self.fc1 = nn.Linear(self.h_size_caller + self.h_size_callee + self.h_size_args, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)
    
    def forward(self, caller, callee, args):

        h_caller = self.gru_caller(self.embedding(caller))
        h_callee = self.gru_callee(self.embedding(callee))
        h_args   = self.gru_args(self.embedding_args(args)) 
        print("h_caller:", h_caller.size(), "h_callee:", h_callee.size(), "h_args:", h_args.size())
        concat  = torch.cat((h_caller, h_callee, h_args), 1)
        print("concat:", concat.size())
        h_fc1 = F.relu(self.fc1(concat))
        h_fc2 = F.relu(self.fc2(h_fc1))
        h_fc3 = F.relu(self.fc3(h_f2))
        output = F.softmax(self.fc4(h_fc3))
        return output
    def forward(self, inp, hidden):
        return self.gru(self.embedding(inp), hidden)
    
    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))

###########################################################################
# class UberNet(nn.Module):                                               #
#     def __init__(self, input_dim = 15, hidden_dim = 10, type = "LSTM"): #
#         super(RNN, self).__init__()                                     #
#         self.INPUT_DIM = input_dim                                      #
#         if type=="LSTM":                                                #
#             self.rnn = nn.LSTM(input_dim, hidden_dim)                   #
#         self.fc1 = nn.Linear(hidden_dim, 10, bias = True)               #
#         self.fc2 = nn.Linear(10, 1, bias = True)                        #
#                                                                         #
#                                                                         #
#     def create_emb_layer(weights_matrix, non_trainable=False):          #
#         num_embeddings, embedding_dim = weights_matrix.size()           #
#         emb_layer = nn.Embedding(num_embeddings, embedding_dim)         #
#         emb_layer.load_state_dict({'weight': weights_matrix})           #
#         if non_trainable:                                               #
#             emb_layer.weight.requires_grad = False                      #
#                                                                         #
#         return emb_layer, num_embeddings, embedding_dim                 #
#                                                                         #
#     def forward(self, inputs):                                          #
#                                                                         #
#                                                                         #
#         _, (last_h, last_c) = self.rnn(inputs)                          #
#         fc1 = F.relu(self.fc1(last_h))                                  #
#         prediction = self.fc2(fc1)                                      #
#         return prediction                                               #
###########################################################################
