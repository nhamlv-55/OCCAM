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
from grpc_server import QueryOracleServicer, Mode, serve_multiple
import multiprocessing
torch.manual_seed(1)
np.random.seed(1)

debug_print_limit = 6
lr = 0.001
minimize = True
MIN_USABLE_RUNS = 15
PLOT_ITER = 2
class PolicyGradient(BasePolicy):
    def __init__(self, workdir, model_path, network_type, network_hp, grpc_mode, debug, metric):
        BasePolicy.__init__(self, workdir, model_path, network_type, network_hp, grpc_mode, debug, metric)
        if network_hp is not None:
            self.net = network_type(self.metadata, network_hp)
        else:
            self.net = network_type(self.metadata)
        #self.atomizer = Atomizer()

    def train(self, model_path, no_of_sampling, no_of_iter, from_scratch):
        collect_encoded_state = False
        if from_scratch:
            print("Create a new model")
            self.save_model(model_path)
        else:
            self.net = torch.load(model_path)
        self.optimizer = optim.Adam(self.net.parameters(), lr = lr)

        for i in range(no_of_iter):
            start_time = time.time()
            if (i+1)%PLOT_ITER == 0:
                fig_name = self.metadata["metric"].replace(" ", "_")
                self.plot(no_of_sampling, i, fig_name)
            eps_threshold = -1 #set to -1 to always use policy
            if self.grpc_mode:
                if self.net.net_type == "UberNet":
                    eps_threshold = -200 #to trigger rnn state in MLPolicy
                    collect_encoded_state = True
                    workers = serve_multiple(15, Mode.TRAINING_RNN, 1, 1, self.workdir, self.net)
                    time.sleep(2) #make sure all models and dicts are loaded into grpc_server
                    self.run_policy(no_of_sampling, eps_threshold, i)
                    time.sleep(1) #make sure all clients have finished
                    for w in workers:
                        w.terminate()
                        if self.debug: print("%s is terminated"%str(w))
                    print("all grpc workers are terminated")
                else:
                    workers = serve_multiple(20, Mode.TRAINING, 1, 1, self.workdir, self.net)
                    time.sleep(1) #make sure all models and dicts are loaded into grpc_server
                    self.run_policy(no_of_sampling, eps_threshold, i)
                    time.sleep(1) #make sure all clients have finished
                    for w in workers:
                        w.terminate()
                        if self.debug: print("%s is terminated"%str(w))
                    print("all grpc workers are terminated")
#                    Previrt_pb2_grpc.add_QueryOracleServicer_to_server(
#                        QueryOracleServicer(mode = Mode.TRAINING_RNN, workdir = self.workdir, atomizer = self.atomizer, net = self.net, debug = False), server)
#                else:
#                    Previrt_pb2_grpc.add_QueryOracleServicer_to_server(
#                        QueryOracleServicer(mode = Mode.TRAINING, workdir = self.workdir, atomizer = self.atomizer, net = self.net, debug = True), server)
#                server.add_insecure_port('[::]:50051')
#                server.start()
#                self.run_policy(no_of_sampling, eps_threshold, i)
#                server.stop(0)
            else:
                self.run_policy(no_of_sampling, eps_threshold, i)
            run_policy_time = time.time()
            print("Rollout %s runs in %s seconds"%(no_of_sampling, run_policy_time - start_time))
            dataset = Dataset(self.dataset_path, collect_encoded_state = collect_encoded_state, size = no_of_sampling)
            #abort bad runs
            if dataset.no_good_runs < MIN_USABLE_RUNS:
                print("too many broken runs")
                continue
            trajectory_data = dataset.get_trajectory_data(metric = self.metadata["metric"], normalize_rewards = True)
            collect_data_time = time.time()
            print("Processing data in ", collect_data_time - run_policy_time)
            self.optimize(trajectory_data, i)
            optimize_time = time.time()
            print("Optimize in ", optimize_time  - collect_data_time)
            if not self.grpc_mode:
                self.save_model(model_path)
            

    def optimize(self, trajectory_data, iteration):
        batch_states = trajectory_data[0]
        batch_rnn_states = trajectory_data[1]
        batch_actions = trajectory_data[2]
        batch_rewards = trajectory_data[3]
        self.optimizer.zero_grad()
        rnn_state_tensor = torch.tensor(batch_rnn_states)
        state_tensor = torch.tensor(batch_states)
        reward_tensor = torch.tensor(batch_rewards)
        # Actions are used as indices, must be LongTensor
        action_tensor = torch.LongTensor(batch_actions)

        print("state_tensor:", state_tensor.shape, "reward_tensor:", reward_tensor.shape, "action_tensor:", action_tensor.shape)

        # Calculate loss
        if self.net.net_type == "UberNet":
            #check rnn_state_tensor
            if rnn_state_tensor.shape[-1]==0:
                #print("ERROR: broken!")
                #print(trajectory_data)
                #do not optimize
                return

            #print("piece of rnn_state", rnn_state_tensor[:debug_print_limit])
            predict_tensor = self.net.forward(rnn_state_tensor)
        else:
            predict_tensor = self.net.forward(state_tensor)
        logprob = torch.log(predict_tensor).view(-1,2).double()
        if self.debug: print("state_tensor:", state_tensor[:debug_print_limit])
        if self.debug: print("predict_tensor:", predict_tensor[:debug_print_limit])
        #if self.debug: print("logprob:", logprob)
        if self.debug: print("reward_tensor", reward_tensor[:debug_print_limit])
        if self.debug: print("action_tensor", action_tensor[:debug_print_limit])
        adv = logprob[np.arange(len(action_tensor)), action_tensor]

        if self.debug:
            print("adv", adv[:debug_print_limit], adv.shape)
        selected_logprobs = reward_tensor * adv
        if self.debug: print("selected_logprobs", selected_logprobs[:debug_print_limit])
        if minimize:
            loss = selected_logprobs.mean()
        else:
            loss = -selected_logprobs.mean()
        print("loss at iteration %s:"%iteration, loss)
        # Calculate gradients
        loss.backward()
        if self.debug:
            print("before updating parameters-------------")
            for param in self.net.parameters():
                print("params:", param.data.view(1, -1),)
        
        # Apply gradients
        self.optimizer.step()
        if self.debug:
            print("after updating parameters---------------")
            for param in self.net.parameters():
                print("grads:", param.grad.view(1, -1))
            for param in self.net.parameters():
                print("params:", param.data.view(1, -1))
    def forward(self, state):
        state_tensor = torch.tensor(state)
        return self.net.forward((state_tensor)).view(-1, 2)

    def plot(self,no_of_sampling, iteration, fig_name ):
        metric = self.metadata["metric"]
        plot(no_of_sampling, iteration, self.workdir, fig_name, self.metadata["agg_results"][metric], self.metadata["none_results"][metric], self.metadata["metric"] )

    def evaluate(self, tag="eval"):
        if self.grpc_mode:
             server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
             Previrt_pb2_grpc.add_QueryOracleServicer_to_server(
                 QueryOracleServicer(self.net, debug = True), server)
             server.add_insecure_port('[::]:50051')
             server.start()
             _ = subprocess.check_output(("%s -epsilon 0 -folder %s 2>%s.log"%(self.run_command, tag, tag)).split(), cwd = self.workdir)
             server.stop(0)
        else:
             _ = subprocess.check_output(("%s -epsilon 0 -folder %s 2>%s.log"%(self.run_command, tag, tag)).split(), cwd = self.workdir)
