import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Dataset
from net import FeedForward, RNN, neural_net
import os
from sklearn import preprocessing
import torch.optim as optim
import argparse

# LSTM in pytorch expects input in form (seq_len, batch_size, input_size), hence we use view(len, 1, -1)
# Before getting to the example, note a few things. Pytorch's LSTM expects all of its inputs to be 3D tensors.
# The semantics of the axes of these tensors is important.
# The first axis is the sequence itself,
# the second indexes instances in the mini-batch,
# and the third indexes elements of the input.
# We havent discussed mini-batching, so lets just ignore that and assume we will always have just 1
# dimension on the second axis. If we want to run the sequence model over the sentence The cow jumped, our input should look like
# |The  [0.3, 0.4, ...] |
# |cow  [0.2, 0.5, ...] |
# |jumped[.7, 0.8, ...]|
# Except remember there is an additional 2nd dimension with size 1.


DEBUG = False
OCCAM_HOME = os.environ['OCCAM_HOME']
model_path = os.path.join(OCCAM_HOME, "razor/MLPolicy/model")

parser = argparse.ArgumentParser()
parser.add_argument('-dataset_path', default=os.path.join(OCCAM_HOME, "examples/portfolio/tree/slash/"), help='s')


args = parser.parse_args()
dataset_path = args.dataset_path
print("dataset_path=%s"%dataset_path)
#load latest model or create a new one
if not os.path.exists(model_path):
    print("No existing model. Create a new one.")
    net = neural_net("RNN")
    for params in net.parameters():
        print(params)
    torch.save(net, model_path)
    inputs = [torch.rand(1, net.INPUT_DIM, dtype=torch.float)]
    inputs = torch.cat(inputs).view(len(inputs), 1, -1)
    print("example inputs:", inputs)
    trial = net.forward(inputs)
    print(trial)
    traced_script_module = torch.jit.trace(net, inputs)

    traced_script_module.save("model.pt")
    quit()
else:
    print("Found an existing model. Load %s"%model_path)
    net = torch.load(model_path)

dataset = Dataset(dataset_path)
X_train, X_test, Y_train, Y_test = dataset.split_dataset()
print(X_train[0])
print(Y_train[0])
#normalize input
#min_max_scaler = preprocessing.MinMaxScaler()
#X = min_max_scaler.fit_transform(X)


#trial run
print(X_train[0].size())
Y_pred = net(X_train[0])
print("trial running: \n", Y_pred)

loss_function = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.1)

for i in range(400):
    for j in range(len(X_train)):
        optimizer.zero_grad()   # zero the gradient buffers
        Y_pred = net(X_train[j])
        loss = loss_function(Y_pred, Y_train[j])
        if i%100==0:
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
traced_script_module = torch.jit.trace(net, X_train[0])

traced_script_module.save("model.pt")


