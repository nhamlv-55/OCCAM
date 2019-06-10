import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Dataset
from net import Net
import os

from sklearn import preprocessing
import torch.optim as optim
DEBUG = False

model_path = "/Users/e32851/workspace/OCCAM/razor/MLPolicy/model"

#load latest model or create a new one
if not os.path.exists(model_path):
    print("No existing model. Create a new one.")
    net = Net()
    for params in net.parameters():
        print(params)
    torch.save(net, model_path)
    example = torch.rand(1, 3)
    traced_script_module = torch.jit.trace(net, example)

    traced_script_module.save("model.pt")
    quit()
else:
    print("Found an existing model. Load %s"%model_path)
    net = torch.load(model_path)

dataset = Dataset("/Users/e32851/workspace/OCCAM/examples/portfolio/tree/slash/", no_of_feats = 3).all_data[0]
print(dataset["input"][:4])
print(dataset["output"][:4])
print("best score for this batch: ", dataset["score"])
X = dataset["input"]
#normalize input
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

X = torch.FloatTensor(X)
print(X)
Y_target = torch.FloatTensor(dataset["output"])

#trial run
Y_pred = net.forward(X)
print("trial running: \n", Y_pred)


criterion = nn.MSELoss()
# create your optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)

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


#print out the model for debugging purpose
print("Final model")
for params in net.parameters():
    print(params)
torch.save(net, model_path)
example = torch.rand(1, 3)
traced_script_module = torch.jit.trace(net, example)

traced_script_module.save("model.pt")


