import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Dataset
import os

import torch.optim as optim
DEBUG = False


model_path = "/Users/e32851/workspace/OCCAM/razor/MLPolicy/model"

dataset = Dataset("/Users/e32851/workspace/OCCAM/examples/portfolio/tree/slash/", no_of_feats = 3).all_data[0]
print(dataset["input"][:4])
print(dataset["output"][:4])
print("best score for this batch: ", dataset["score"])
X = torch.FloatTensor(dataset["input"])
Y = torch.FloatTensor(dataset["output"])
print(X.size())
print(Y.size())

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 3)  # 6*6 from image dimension 
        self.fc2 = nn.Linear(3, 2)
        

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(x, dim = 1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

#load latest model
if not os.path.exists(model_path):
    print("No existing model. Create a new one.")
    net = Net()
else:
    print("Found an existing model. Load %s"%model_path)
    net = torch.load(model_path)

Y_target = Y

criterion = nn.MSELoss()
# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

for i in range(400):
# in your training loop:
    optimizer.zero_grad()   # zero the gradient buffers
    Y_pred = net(X)
    loss = criterion(Y_pred, Y_target)
    if i%20==0:
        print(loss)
        if DEBUG:
            print(Y_pred)
            for params in net.parameters():
                print(params)
    loss.backward()
    optimizer.step()    # Does the update
torch.save(net, model_path)
example = torch.rand(1, 3)
traced_script_module = torch.jit.trace(net, example)

traced_script_module.save("model.pt")


