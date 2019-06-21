import torch.nn as nn
import torch.nn.functional as F
import torch
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.INPUT_DIM = 14
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
