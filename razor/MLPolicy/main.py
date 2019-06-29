from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Dataset
from net import FeedForward, RNN, neural_net
import os
from sklearn import preprocessing
import torch.optim as optim
import argparse
import subprocess
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
parser.add_argument('-workdir', default=os.path.join(OCCAM_HOME, "examples/portfolio/tree/"), help='s')
parser.add_argument('-action', default="bootstrap")

args = parser.parse_args()
workdir = args.workdir
dataset_path = os.path.join(workdir, "slash")
action = args.action
print("dataset_path=%s"%dataset_path)
#load latest model or create a new one

def bootstrap(model_path):
    net = neural_net("FeedForwardSingleInput")
    for params in net.parameters():
        print(params)
    torch.save(net, model_path)
    inputs = torch.rand(2, net.features_dim, dtype=torch.float)
    print("example inputs:", inputs, inputs.size())
    trial = net.forward(inputs)
    print(trial)
    traced_script_module = torch.jit.trace(net, inputs)
    traced_script_module.save("model.pt")
    return net

def loss_function(history, reward):
    product = torch.mul(history, reward)
    print("product:", product)
    negation = product.mul(-1)
    print("negation:", negation)
    loss = torch.sum(negation, -1)
    print("loss:", loss)
    return loss

def train(model_path, no_of_sampling = 3, no_of_iter = 3):
    if not os.path.exists(model_path):
        print("No existing model. Create a new one.")
        net = bootstrap(model_path)
    else:
        net = torch.load(model_path)

    # For debugging only
    loss_stack = []
    # Define optimizer
    optimizer = optim.Adam(net.parameters(), 
                           lr=0.01)
    for i in range(no_of_iter):
        #use parallel to run slash
        if os.path.exists(dataset_path):
            clear_prev_runs = subprocess.check_output(("rm -r %s"%dataset_path).split())
        job_ids = ""
        for j in range(no_of_sampling):
            job_ids+=" %s"%str(j)
        runners_cmd = "parallel ./build.sh --disable-inlining --devirt none -folder {} ::: %s"%job_ids
        print(runners_cmd)
        runners = subprocess.check_output(runners_cmd.split(), cwd = workdir)
        dataset = Dataset(dataset_path, size = no_of_sampling)
        batch_states, batch_actions, batch_rewards, batch_probs = dataset.get_run_data()
        print("batch_states:", batch_states)
        print("batch_actions:", batch_actions)
        print("batch_rewards:", batch_rewards)
        print("batch_probs:", batch_probs)
        optimizer.zero_grad()
        state_tensor = torch.FloatTensor(batch_states)
        reward_tensor = torch.FloatTensor(batch_rewards)
        # Actions are used as indices, must be LongTensor
        action_tensor = torch.LongTensor(batch_actions)
        prob_tensor = net.forward(state_tensor)
        print("prob_tensor:", prob_tensor)
        print("batch_probs:", batch_probs)
        print("Check if the 2 above tensors are the same")
        # Calculate loss
        logprob = torch.log(prob_tensor)
        print(logprob)
        print(np.arange(len(action_tensor)))
        print(action_tensor)
        selected_logprobs = reward_tensor * \
            logprob[np.arange(len(action_tensor)), action_tensor]
        print(selected_logprobs)

        loss = selected_logprobs.mean()
        loss_stack.append(loss)
        
        # # Calculate gradients
        loss.backward()
        # # Apply gradients
        optimizer.step()

        # batch_rewards = []
        # batch_actions = []
        # batch_states = []
        # batch_counter = 1

        print(loss_stack)









def train_old(model_path):
    if not os.path.exists(model_path):
        print("No existing model. Create a new one.")
        net = bootstrap(model_path)
    else:
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
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    for i in range(0):
        for j in range(len(X_train)):
            optimizer.zero_grad()   # zero the gradient buffers
            Y_pred = net(X_train[j])
            loss = loss_function(Y_pred, Y_train[j])
            if i%100==0:
                print("---------------------")
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


if __name__=="__main__":
    if action=="bootstrap":
        bootstrap(model_path)
    elif action=="train":
        train(model_path)

