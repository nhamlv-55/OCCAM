from __future__ import print_function
import os
import glob
import numpy as np
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import random
from collections import namedtuple
import json
import subprocess
import argparse
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
Step   = namedtuple('Step', ('state', 'prob', 'action'))
np.set_printoptions(precision=6, suppress=True)
GAMMA = 0.99
DEBUG = False
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.eps_reward = []
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Dataset(object):
    def __init__(self, folder, n_unused_stat = 3, size=99999999):
        self.folder = folder
        print(self.folder)
        self.n_unused_stat = n_unused_stat
        self.all_data = []
        self.collect(size)
        self.calculate_std_mean()
        self.features_len = self.mean.shape[0] - n_unused_stat
        self.mean = self.mean[:self.features_len]
        self.std  = self.std[:self.features_len]

    def merge_csv(self, csv_files):
        episode_data = []
        raw_data  = []
        #print(csv_files)
        input = []
        output = []
        total = 0
        for fname in csv_files:
            run = []
            with open(fname, "r") as f:
                for l in f.readlines():
                    if l.startswith("TOUCH A CALL"):
                        continue
                    total+=1
                    tokens = l.strip().split(',')
                    tokens = [float(t) for t in tokens]
                    raw_data.append(tokens)
                    state = tokens[:-self.n_unused_stat]
                    prob = tokens[-3:-1]
                    action = tokens[-1]
                    run.append(Step(state, prob, action))
            if len(run)>0:
                episode_data.append(run)
        return raw_data, episode_data, total

    def get_stat(self, run):
        result = {}
        with open(run+"/0_results.txt", "r") as f:
            for l in f.readlines():
                if "[" in l:
                    continue
                tokens = l.strip().split()
                v = int(tokens[0])
                k = " ".join(tokens[1:])
                result[k] = v
        return result
    
    def score(self, result):
        return result["Number of instructions"]
        #return 1.0*result["Statically safe memory accesses"]/result["Number of memory instructions"]
        #return result["Statically unknown memory accesses"]
    def get_step_reward(self, current_state, next_state, final_score):
        #reward is 0 for all step
        if next_state is not None:
            return float(0)
        else:
            return float(final_score)

       # if next_state is not None:
       #     print("next_state is not None")
       #     return next_state[0][18]-current_state[0][18]
       # else:
       #     print("next_state is None")
       #     print("final score:", final_score)
       #     return final_score-current_state[0][18]

    def calculate_std_mean(self):
        raw_data_np = np.array(self.raw_data)
        print(len(raw_data_np))
        self.mean = np.mean(raw_data_np, 0)
        self.std  = np.std(raw_data_np, 0)
        #calculate mean and std of scores based on current dataset
        final_scores = []
        for eps in self.all_data:
            final_scores.append(eps["score"])
        final_scores = np.array(final_scores)
        self.score_mean = np.mean(final_scores, 0)
        self.score_std  = np.std(final_scores, 0)

    def collect(self, size):
        self.raw_data = []
        runs = glob.glob(self.folder+"/run*")
        sorted(runs)
        size = min(len(runs), size)
        for r in runs[:size]:
            run_data = {}
            #print(r)
            csv_files = glob.glob(r+"/*.csv")
            _, episode_data, total = self.merge_csv(csv_files)
            result = self.get_stat(r)
            self.raw_data.extend(_)
            run_data["episode_data"] = episode_data
            run_data["score"] = self.score(result)
            #run_data["discounted_r"] = self.discounted_rewards(run_data["score"])
            run_data["raw_result"] = result
            run_data["total"] = total
            self.all_data.append(run_data)

    def sort(self):
        self.all_data.sort(key=lambda x: x["score"])

    # for Policy Gradient
    def get_trajectory_data(self):
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_probs = []
        for eps in self.all_data:
            states = []
            actions = []
            probs = []
            
            for sub_episode in eps["episode_data"]:
                for step in sub_episode:
                    states.append(step.state)
                    actions.append(step.action)
                    probs.append(step.prob)
                
            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_probs.extend(probs)
            rewards = [eps["score"]]*len(states)
            batch_rewards.extend(discounted_rewards(rewards, GAMMA))
        return batch_states, batch_actions, batch_rewards, batch_probs 
    def push_to_memory(self, memory):

        ##################################################
        # action = select_action(state)                  #
        # _, reward, done, _ = env.step(action.item())   #
        # reward = torch.tensor([reward], device=device) #
        #                                                #
        # # Observe new state                            #
        # last_screen = current_screen                   #
        # current_screen = get_screen()                  #
        # if not done:                                   #
        #     next_state = current_screen - last_screen  #
        # else:                                          #
        #     next_state = None                          #
        #                                                #
        # # Store the transition in memory               #
        # memory.push(state, action, next_state, reward) #
        ##################################################

        for eps in self.all_data:
            for i in range(len(eps["episode_data"])):
                #print("run %s"%str(i))
                sub_episode = eps["episode_data"][i]
                
                if i==len(eps["episode_data"])-1:
                    next_sub_episode = (None, None, None)
                else:
                    next_sub_episode = eps["episode_data"][i+1]
                for j in range(len(sub_episode)):
                    step = sub_episode[j]
                    #print(step)
                    if j==len(sub_episode)-1:
                        next_step = next_sub_episode[0]
                        #next_level = i+1
                    else:
                        next_step = sub_episode[j+1]
                        #next_level = i
                    state = torch.tensor(step.state).view(1, -1)
                    action = torch.tensor(step.action, dtype=torch.long).view(1,1)
                    if next_step is not None:
                        next_state = torch.tensor(next_step.state).view(1, -1)
                        #next_state.append(next_level)
                    else:
                        next_state = None

                    reward = torch.tensor([self.get_step_reward(state, next_state, eps["score"])])
                    #if reward!=0:
                    #    print(state, action, next_state, reward, "<<<")
                    memory.push(state, action, next_state, reward)
        print("last entry in memory:", memory.memory[-1])

    

    def dump(self):
        ##########################################################
        # for r in self.all_data:                                #
        #     print("score:", r["score"])                        #
        #     print("number of call sites:", r["total"])         #
        #     print("number of passes:", len(r["episode_data"])) #
        #     for run in r["episode_data"]:                      #
        #         for callsite in run:                           #
        #             print(callsite)                            #
        ##########################################################
        #        print("------")
        print("best score", self.all_data[0]["score"])
        print("worst score", self.all_data[-1]["score"])
def discounted_rewards(rewards, gamma):
    r = np.array([gamma**i * rewards[i] 
                  for i in range(len(rewards))])
    if DEBUG: print("r:", r)
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    if DEBUG: print("r cumsum, mean, std:", r)
    r -= np.mean(r)
    r /= np.std(r)
    if DEBUG: print("scaled_dr:", r)
    return r
def gen_new_meta(workdir, bootstrap_runs, run_command):
    dataset_path = os.path.join(workdir, "slash")
    metadata = {}
    # run slash without the model to see how many features we are using
    print("run check_format to see if we change the number of features")
    #clear previous runs
    if os.path.exists(dataset_path):
        clear_prev_runs = subprocess.check_output(("rm -rf %s"%dataset_path).split())
    #run with policy none
    run_none = subprocess.check_output("./build.sh -intra-spec none -folder none".split(), cwd = workdir)
    #run with policy aggressive
    run_aggr = subprocess.check_output("./build.sh -intra-spec nonrec-aggressive -folder agg".split(), cwd = workdir)
    #build 1 2 3 4 ... k
    job_ids = ""
    for jid in range(bootstrap_runs):
        job_ids +=" %s"%str(jid)
    #run the jobs
    runners_cmd = "parallel %s -epsilon 10.5 -folder {} 2>/dev/null ::: %s"%(run_command, job_ids)
    print(runners_cmd)
    runners = subprocess.check_output(runners_cmd.split(), cwd = workdir)

    dataset_bootstrap = Dataset(dataset_path)
    dataset_bootstrap.sort()
    dataset_bootstrap.dump()
    print("features_len:", dataset_bootstrap.features_len)
    print("mean:", dataset_bootstrap.mean)
    print("std:", dataset_bootstrap.std)
    metadata["features_len"] = dataset_bootstrap.features_len
    metadata["mean"] = dataset_bootstrap.mean.tolist()
    metadata["std"] = dataset_bootstrap.std.tolist()
    metadata["score_mean"] = dataset_bootstrap.score_mean
    metadata["score_std"]  = dataset_bootstrap.score_std
    metadata["sample_inputs"] = dataset_bootstrap.raw_data[0][:dataset_bootstrap.features_len]
    with open(os.path.join(workdir, "metadata.json"), "w") as f:
        json.dump(metadata, f)


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', default=100, help='number of trial runs to get meta data')
    args = parser.parse_args()
    bootstrap_runs = int(args.n)
    OCCAM_HOME = os.environ['OCCAM_HOME']
    model_path = os.path.join(OCCAM_HOME, "razor/MLPolicy/model") 
    work_dir   = os.path.join(OCCAM_HOME, "examples/portfolio/tree")
    gen_new_meta(work_dir, bootstrap_runs, "./build.sh ")
