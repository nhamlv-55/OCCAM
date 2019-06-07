import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Dataset

dataset = Dataset("/Users/e32851/workspace/OCCAM/examples/portfolio/tree/slash/", no_of_feats = 3).all_data[0]
print(dataset["input"][:4])
print(dataset["output"][:4])
X = torch.FloatTensor(dataset["input"])
Y = torch.LongTensor(dataset["output"])
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


net = Net()
print(net)

Y_target = Y

criterion = nn.CrossEntropyLoss()

import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

for i in range(400):
# in your training loop:
    optimizer.zero_grad()   # zero the gradient buffers
    Y_pred = net(X)
    loss = criterion(Y_pred, Y_target)
    if i%20==0:
        print(loss)
    loss.backward()
    optimizer.step()    # Does the update

example = torch.rand(1, 3)
traced_script_module = torch.jit.trace(net, example)

traced_script_module.save("model.pt")


