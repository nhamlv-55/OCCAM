from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from net import * 
from PolicyGradient import PolicyGradient
from DAgger import DAgger
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
def evaluate(model_path):
    _ = subprocess.check_output("./build.sh -epsilon 0 -folder eval 2>eval.log".split(), cwd = workdir)
    print(_)
    print("done evaluation")
if __name__=="__main__":
    if action=="gen-meta":
        gen_new_meta()
    elif action=="train-scratch":
        #policy = DoubleQPolicy(workdir, model_path, FeedForwardSingleInput, network_hp = None)
        policy = PolicyGradient(workdir, model_path, FeedForwardSingleInputSoftmax, network_hp = None, grpc_mode = True)
        #policy = DAgger(workdir, model_path, FeedForwardSingleInputSoftmax, network_hp = None)
        policy.train(model_path, no_of_sampling, no_of_iter, from_scratch = True)
    elif action=="train-continue":
        policy = DoubleQPolicy(workdir, model_path, FeedForwardSingleInput, network_hp = None)
        policy.train(model_path, no_of_sampling, no_of_iter, from_scratch = False)
    elif action=="eval":
        evaluate(model_path)
