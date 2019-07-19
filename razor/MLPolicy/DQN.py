from __future__ import print_function
import torch
from net import *
from basePolicy import BasePolicy
import os
import subprocess
import math
from utils import *
import torch.optim as optim
BATCH_SIZE = 1000
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 5
DEBUG = False
class DQNPolicy(BasePolicy):
    def __init__(self, workdir, model_path, network_type, network_hp):
        BasePolicy.__init__(self, workdir, model_path, network_type, network_hp)
        if network_hp is not None:
            self.net = network_type(self.metadata, network_hp)
            self.target_net = network_type(self.metadata, network_hp)
        else:
            self.net = network_type(self.metadata)
            self.target_net = network_type(self.metadata)
        self.target_net.load_state_dict(self.net.state_dict())

    def train(self, model_path, no_of_sampling, no_of_iter, from_scratch):
        if from_scratch:
            print("Create a new model")
            self.save_model(model_path)
        else:
            self.net = torch.load(model_path)
        self.optimizer = optim.RMSprop(self.net.parameters())

        for i in range(no_of_iter):
            if (i+1)%10==0:
                print("performance at iteration %s"%str(i))
                self.evaluate(tag="eval%s"%str(i))
            steps_done = i
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
            self.run_policy(no_of_sampling, eps_threshold)
            dataset = Dataset(self.dataset_path, size = no_of_sampling)
            dataset.push_to_memory(self.memory)
            if DEBUG: print(self.memory.memory)
            self.optimize()
            self.save_model(model_path)
            # Update the target network
            if i% TARGET_UPDATE == 1:
                print("Update target network...")
                self.target_net.load_state_dict(self.net.state_dict())

    def optimize(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))
        if DEBUG: print("batch:", batch)
        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        #print("batch.reward:", batch.reward)
        reward_batch = torch.cat(batch.reward)

        if DEBUG: print("state_batch 0:",state_batch[0])
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.net(state_batch).gather(1, action_batch)
        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        # try using direct values
        #expected_state_action_values = reward_batch
        torch.set_printoptions(sci_mode = False)
        if DEBUG: print("next_state_values:", next_state_values)
        if DEBUG: print("state_action_values:", state_action_values)
        if DEBUG: print("expected_state_action_values", expected_state_action_values)
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        print("loss", loss)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.net.parameters():
            #print(param)
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
