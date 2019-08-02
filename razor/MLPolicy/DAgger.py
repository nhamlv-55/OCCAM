from __future__ import print_function
import torch
from net import *
from basePolicy import BasePolicy
import os
import subprocess
import math
from utils import *
import torch.optim as optim
DEBUG = False
TOP_K_PERC = .15 #using 15% best sample to put to DAgger
EPOCH = 100
class DAgger(BasePolicy):
    def __init__(self, workdir, model_path, network_type, network_hp):
        BasePolicy.__init__(self, workdir, model_path, network_type, network_hp)
        if network_hp is not None:
            self.net = network_type(self.metadata, network_hp)
        else:
            self.net = network_type(self.metadata)

        self.agg_dataset = {}
        self.loss = torch.nn.NLLLoss()
    def train(self, model_path, no_of_sampling, no_of_iter, from_scratch):
        self.no_of_best_traj = int(no_of_sampling*TOP_K_PERC)
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
            self.run_policy(no_of_sampling, eps_threshold)
            dataset = Dataset(self.dataset_path, size = no_of_sampling)
            dataset.sort()
            top_traj = dataset.all_data[:self.no_of_best_traj]
            print("best score of previous batch:", top_traj[0]["score"])
            if DEBUG: print("top_traj:", top_traj, len(top_traj))
            self.aggregate_data(top_traj)
            self.optimize(i)
            self.save_model(model_path)

    def aggregate_data(self, top_traj):
        for traj in top_traj:
            for sub_episode in traj["episode_data"]:
                for step in sub_episode:
                    state = tuple(step.state)
                    action = int(step.action)
                    if state in self.agg_dataset:
                        self.agg_dataset[state][action]+=1
                    else:
                        self.agg_dataset[state]=[0,0]
                        self.agg_dataset[state][action]+=1
        if DEBUG:
            for state in self.agg_dataset:
                print(state, self.agg_dataset[state])

    
    def optimize(self, iteration):
        batch_states = []
        batch_actions = []
        for state in self.agg_dataset:
            batch_states.append(state)

            batch_actions.append(np.argmax(self.agg_dataset[state]))

        for i in xrange(EPOCH):
            self.optimizer.zero_grad()
            state_tensor = torch.tensor(batch_states)
            action_tensor = torch.tensor(batch_actions)
            # Calculate loss
            prediction = torch.log(self.net.forward(state_tensor))
            if DEBUG: print("prediction:", prediction)
            if DEBUG: print("action_tensor:", action_tensor)
            loss= self.loss(prediction, action_tensor)
            print("loss at EPOCH %s of iteration %s: \n\t"%(i, iteration), loss)
            # Calculate gradients
            loss.backward()
            # Apply gradients
            self.optimizer.step()
