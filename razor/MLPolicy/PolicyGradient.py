from __future__ import print_function
import torch
from net import *
from basePolicy import BasePolicy
import os
import subprocess
import math
from utils import *
import torch.optim as optim
from concurrent import futures
import time
import math
import logging

import grpc

import Previrt_pb2
import Previrt_pb2_grpc
from grpc_server import QueryOracleServicer
DEBUG = False
class PolicyGradient(BasePolicy):
    def __init__(self, workdir, model_path, network_type, network_hp, grpc_mode = False, debug = False):
        BasePolicy.__init__(self, workdir, model_path, network_type, network_hp, grpc_mode, debug)
        if network_hp is not None:
            self.net = network_type(self.metadata, network_hp)
        else:
            self.net = network_type(self.metadata)
            
    def train(self, model_path, no_of_sampling, no_of_iter, from_scratch):
        if from_scratch:
            print("Create a new model")
            self.save_model(model_path)
        else:
            self.net = torch.load(model_path)
        self.optimizer = optim.Adam(self.net.parameters(), lr = 0.01)

        for i in range(no_of_iter):
            if (i+1)%10 == 0:
                print("performance at iteration %s"%str(i))
                self.evaluate(tag="eval%s"%str(i))
            eps_threshold = -1 #set to -1 to always use policy
            if self.grpc_mode:
                server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
                Previrt_pb2_grpc.add_QueryOracleServicer_to_server(
                    QueryOracleServicer(self.net), server)
                server.add_insecure_port('[::]:50051')
                server.start()
                self.run_policy(no_of_sampling, eps_threshold)
                server.stop(0)
            else:
                self.run_policy(no_of_sampling, eps_threshold)
            dataset = Dataset(self.dataset_path, size = no_of_sampling)
            trajectory_data = dataset.get_trajectory_data()
            self.optimize(trajectory_data, i)
            self.save_model(model_path)

    def optimize(self, trajectory_data, iteration):
        batch_states = trajectory_data[0]
        batch_actions = trajectory_data[1]
        batch_rewards = trajectory_data[2]
        self.optimizer.zero_grad()
        state_tensor = torch.tensor(batch_states)
        reward_tensor = torch.tensor(batch_rewards)
        # Actions are used as indices, must be LongTensor
        action_tensor = torch.LongTensor(batch_actions)

        print(state_tensor.shape, reward_tensor.shape, action_tensor.shape)
        # Calculate loss
        logprob = torch.log(
            self.net.forward(state_tensor)).view(-1, 2)
        if self.debug: print("state_tensor:", state_tensor)
        if self.debug: print("logprob:", logprob)
        if self.debug: print("reward_tensor", reward_tensor)
        if self.debug: print("action_tensor", action_tensor)
        adv = logprob[np.arange(len(action_tensor)), action_tensor]

        if self.debug:
            print("adv", adv, adv.shape)
        selected_logprobs = reward_tensor * adv
        if self.debug: print("selected_logprobs", selected_logprobs)
        loss = -selected_logprobs.mean()
        print("loss at iteration %s:"%iteration, loss)
        # Calculate gradients
        loss.backward()
        # Apply gradients
        self.optimizer.step()
        if self.debug:
            for param in self.net.parameters():
                print("params:", param.data)

    def forward(self, state):
        state_tensor = torch.tensor(state)
        return self.net.forward((state_tensor)).view(-1, 2)

    def evaluate(self, tag="eval"):
        if self.grpc_mode:
             server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
             Previrt_pb2_grpc.add_QueryOracleServicer_to_server(
                 QueryOracleServicer(self.net, debug = True), server)
             server.add_insecure_port('[::]:50051')
             server.start()
             _ = subprocess.check_output(("%s -epsilon 0 -folder %s 2>%s.log"%(self.run_command, tag, tag)).split(), cwd = self.workdir)
             server.stop(0)
        else:
             _ = subprocess.check_output(("%s -epsilon 0 -folder %s 2>%s.log"%(self.run_command, tag, tag)).split(), cwd = self.workdir)
