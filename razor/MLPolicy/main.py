from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Dataset
from net import FeedForwardSingleInput, RNN, neural_net
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


OCCAM_HOME = os.environ['OCCAM_HOME']
model_path = os.path.join(OCCAM_HOME, "razor/MLPolicy/model")

parser = argparse.ArgumentParser()
parser.add_argument('-workdir', default=os.path.join(OCCAM_HOME, "examples/portfolio/tree/"), help='s')
parser.add_argument('-action', default="bootstrap")
parser.add_argument('-s', default = 10, type=int, help='no of sampling')
parser.add_argument('-i', default = 3, type=int, help ='no of iteration')
parser.add_argument('-d', dest='DEBUG', action = 'store_true')
parser.set_defaults(DEBUG = False)
args = parser.parse_args()
workdir = args.workdir
dataset_path = os.path.join(workdir, "slash")
action = args.action
print("dataset_path=%s"%dataset_path)
no_of_sampling = args.s
no_of_iter = args.i
DEBUG = args.DEBUG
print("DEBUG=", DEBUG)
#load latest model or create a new one
SAMPLE = torch.tensor([41.000000,176.000000,63.000000,0.000000,18.000000,39.000000,1.000000,79.000000,398.000000,98.000000,8.000000,64.000000,78.000000,1.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000])
 
SAMPLE = torch.tensor([4.1,1.76,6.3,0.0,0.18,0.39,1,7.9,3.98,0.98,0.08,0.64,0.78,1.000000, 0, 0])
trace = torch.tensor([0]*42, dtype=torch.float)
print(trace)
SAMPLE = torch.cat((SAMPLE, trace)).view(1, -1)
print("SAMPLE using in trace and bootstrap:", SAMPLE, SAMPLE.size())
def bootstrap(model_path):
    if not os.path.exists(model_path):
        #run OCCAM once to grab the total state
        net = neural_net("FeedForwardSingleInput")
        save_model(net, model_path)
        print("run OCCAM once to grab the total state")
        evaluate(model_path)
        print("bootstrapping again")
        bootstrap(model_path)
    else:
        print("bootstrapping with recorded states")
        dataset = Dataset(dataset_path, size = no_of_sampling)
        batch_states, batch_actions, batch_rewards, batch_probs = dataset.get_run_data()
        total_state = []
        for i in range(21):
            total_state.extend(batch_states[i][:16])
        net = FeedForwardSingleInput(state = total_state )
#    for params in net.parameters():
#        print(params)
    trial = net.forward(SAMPLE)
    print("trial in bootstrap:", trial)
    save_model(net, model_path)
    return net

def save_model(net, model_path):
    torch.save(net, model_path)
    inputs = SAMPLE
    traced_script_module = torch.jit.trace(net, inputs)
    traced_script_module.save(os.path.join(OCCAM_HOME,"model.pt"))

def train(model_path, no_of_sampling, no_of_iter, from_scratch):
    if not os.path.exists(model_path) or from_scratch:
        print("No existing model. Create a new one.")
        net = bootstrap(model_path)
    else:
        net = torch.load(model_path, state = total_state )

    # For debugging only
    loss_stack = []
    # Define optimizer
    optimizer = optim.Adam(net.parameters(), 
                           lr=0.01)
    for i in range(no_of_iter):
        if i%5==1:
            print("performance at iteration %s"%str(i))
            evaluate(model_path)
        #use parallel to run slash
        if os.path.exists(dataset_path):
            clear_prev_runs = subprocess.check_output(("rm -r %s"%dataset_path).split())
        job_ids = ""
        for j in range(no_of_sampling):
            job_ids+=" %s"%str(j)
        runners_cmd = "parallel ./build.sh --disable-inlining --devirt none -folder {} 2>/dev/null  ::: %s"%job_ids
        print(runners_cmd)
        runners = subprocess.check_output(runners_cmd.split(), cwd = workdir)
        dataset = Dataset(dataset_path, size = no_of_sampling)
        batch_states, batch_actions, batch_rewards, batch_probs = dataset.get_run_data()
        
        if DEBUG:
            print("batch_states:", )
            for s in batch_states:
                print(s)
            print("batch_actions:", batch_actions)
            print("batch_rewards:", batch_rewards)
        optimizer.zero_grad()
        state_tensor = torch.FloatTensor(batch_states)
        reward_tensor = torch.FloatTensor(batch_rewards)
        # Actions are used as indices, must be LongTensor
        action_tensor = torch.LongTensor(batch_actions)
        prob_tensor = net.forward(state_tensor)
        trial = net.forward(SAMPLE)
        if DEBUG:
            print("trial:", trial)
            print("prob_tensor:", prob_tensor[:, :10])
            print("batch_probs:", batch_probs[:, :10])
            print("Check if the 2 above tensors are the same")
        # Calculate loss
        logprob = torch.log(prob_tensor)
        if DEBUG:
            print(logprob)
            torch.set_printoptions(profile="full")
            print(action_tensor)
            torch.set_printoptions(profile="default")
            print(np.arange(len(action_tensor)))
        selected_logprobs = reward_tensor * \
            logprob[np.arange(len(action_tensor)), action_tensor]
        print(selected_logprobs)

        loss = selected_logprobs.mean()
        loss_stack.append(loss)
        
        # # Calculate gradients
        loss.backward()
        # # Apply gradients
        optimizer.step()

        #save model
        save_model(net, model_path)

        print(loss_stack)


    #eval after training:
    evaluate(model_path)
def evaluate(model_path):
    _ = subprocess.check_output("./build.sh --disable-inlining --devirt none -folder eval".split(), cwd = workdir)

if __name__=="__main__":
    if action=="bootstrap":
        bootstrap(model_path)
    elif action=="train-scratch":
        train(model_path, no_of_sampling, no_of_iter, from_scratch = True)
    elif action=="train-continue":
        train(model_path, no_of_sampling, no_of_iter, from_scratch = False)
    elif action=="eval":
        evaluate(model_path)
