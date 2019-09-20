from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import pickle
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
        self.net_type = "TinyFFNSoftmax"
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
        self.net_type = "FeedForwardSingleInput"
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
        self.net_type = "FeedForwardSingleInputSoftmax"
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

def create_emb_layer(emb_file, non_trainable = False):
    emb_matrix = np.load(emb_file, allow_pickle = True)
    num_emb, dim_emb = emb_matrix.shape
    print(num_emb)
    print("embedding dtype:", emb_matrix.dtype)
    emb_layer = nn.Embedding(num_emb + 1, dim_emb)
    emb_layer.weight = nn.Parameter(torch.from_numpy(emb_matrix))
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer, num_emb, dim_emb

class UberNet(Net):
    def __init__(self, metadata, dim_hidden = 64, dim_hidden_args = 16, dim_emb_args = 4, num_layers = 2):
        Net.__init__(self, metadata)
        self.extra_feat = ["caller_no_of_use"]
        self.max_sequence_len = metadata["max_sequence_len"]
        self.net_type = "UberNet"
        emb_file = '/home/workspace/OCCAM/razor/MLPolicy/inst2vec/published_results/data/vocabulary/emb.p'
        self.embedding, num_emb, dim_emb = create_emb_layer(emb_file, True)
        self.embedding_args = nn.Embedding(2, dim_emb_args) 
        #print(self.embedding)
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.dim_hidden_args = dim_hidden_args

        #layers
        self.gru_caller = nn.GRU(dim_emb, dim_hidden, num_layers, batch_first = True)
        self.gru_callee = nn.GRU(dim_emb, dim_hidden, num_layers, batch_first = True)
        self.gru_ctx    = nn.GRU(dim_emb, dim_hidden, num_layers, batch_first = True)
        self.gru_args   = nn.GRU(dim_emb_args, dim_hidden_args, 1, batch_first = True)
        self.fc1 = nn.Linear(self.dim_hidden + self.dim_hidden + self.dim_hidden +  self.dim_hidden_args + len(self.extra_feat) , 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
    
    def forward(self, x):
        x = x.long()
        if DEBUG: print("x:", x.size())
        caller_len = x[:, 0]
        callee_len = x[:, 1]
        args_len   = x[:, 2]
        caller_usage = x[:, 3].float()/100
        if DEBUG: print("caller_usage:", caller_usage)
        #print("caller_len:", caller_len, "callee_len:", callee_len, "args_len:", args_len)
        x = x[:, 4:]
        #print("x after cutoff", x.size())
        caller = x[:, :self.max_sequence_len]
        callee = x[:, self.max_sequence_len:2*self.max_sequence_len]
        args   = x[:, 2*self.max_sequence_len:-10]
        ctx    = x[:, -10:]

        if DEBUG: print("caller:", caller)
        if DEBUG: print("callee:", callee)
        if DEBUG: print("args:", args)
        if DEBUG: print("ctx:", ctx)
        packed_caller = nn.utils.rnn.pack_padded_sequence(self.embedding(caller), caller_len, batch_first=True, enforce_sorted = False)
        packed_callee = nn.utils.rnn.pack_padded_sequence(self.embedding(callee), callee_len, batch_first=True, enforce_sorted = False)
        packed_args   = nn.utils.rnn.pack_padded_sequence(self.embedding_args(args), args_len, batch_first=True, enforce_sorted = False)
        #print("packed args",packed_args)
        _, last_h_caller = self.gru_caller(packed_caller.float())
        _, last_h_callee = self.gru_callee(packed_callee.float())
        _, last_h_args   = self.gru_args(packed_args.float())
        _, last_h_ctx    = self.gru_ctx(self.embedding(ctx).float())
        #h_args   = self.gru_args(self.embedding_args(args)) 
        #print("h_caller:", last_h_caller.size(), "h_callee:", last_h_callee.size(), "h_args:", last_h_args.size())
        concat  = torch.cat((last_h_caller[-1], last_h_callee[-1], last_h_args[-1], last_h_ctx[-1], caller_usage.view(-1, 1) ), -1)
        if DEBUG: print("concat:", concat)
        h_fc1 = F.relu(self.fc1(concat))
        h_fc2 = F.relu(self.fc2(h_fc1))
        h_fc3 = F.relu(self.fc3(h_fc2))
        output = F.softmax(h_fc3, dim = -1)
        #print("logit size:", output.size())
        return output
    
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
