import torch
from net import *
from basePolicy import BasePolicy
import os
import subprocess
import math
from utils import Dataset

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

class DQNPolicy(BasePolicy):
    def __init__(self, workdir, model_path, network_type, network_hp):
        BasePolicy.__init__(self, workdir, model_path, network_type, network_hp)
        
    def train(self, model_path, no_of_sampling, no_of_iter, from_scratch):
        if from_scratch:
            print("Create a new model")
        else:
            self.net = torch.load(model_path)
        for i in range(no_of_iter):
            if i%5==1:
                print("performance at iteration %s"%str(i))
                self.evaluate(tag="eval%s"%str(i))
            #clear previous runs
            if os.path.exists(self.dataset_path):
                clear_prev_runs = subprocess.check_output(("rm -r %s"%self.dataset_path).split())
            job_ids = ""
            for jid in range(no_of_sampling):
                job_ids +=" %s"%str(jid)
            steps_done = i*no_of_sampling*self.trace_len
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                            math.exp(-1. * steps_done / EPS_DECAY)
            runners_cmd = "parallel %s -epsilon %s -folder {} 2>/dev/null  ::: %s"%(self.run_command, eps_threshold, job_ids)
            print(runners_cmd)
            runners = subprocess.check_output(runners_cmd.split(), cwd = self.workdir)
            dataset = Dataset(self.dataset_path, size = no_of_sampling)
            dataset.push_to_memory(self.memory)


    def optimize(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
