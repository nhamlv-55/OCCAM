from __future__ import print_function
import os
import glob
import numpy as np
import numpy as np
import torch
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, folder, n_unused_stat = 3, size=-1):
        self.folder = folder
        self.n_unused_stat = n_unused_stat
        self.all_data = []
        self.gamma = 0.99
        self.collect(size)
    def merge_csv(self, csv_files):
        episode_data = []
        raw_data  = []

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
                    key = tokens[:-self.n_unused_stat]
                    prob = tokens[-3:-1]
                    label = tokens[-1]
                    run.append((key, label, prob))
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
        return result["Number of instructions"]*1.0/1000
    

    def collect(self, size):
        runs = glob.glob(self.folder+"/run*")
        sorted(runs)
        for r in runs[:size]:
            run_data = {}
            print(r)
            csv_files = glob.glob(r+"/*.csv")
            _, episode_data, total = self.merge_csv(csv_files)
            result = self.get_stat(r)
            run_data["episode_data"] = episode_data
            run_data["score"] = self.score(result)
            run_data["discounted_r"] = self.discounted_rewards(run_data["score"])
            run_data["raw_result"] = result
            run_data["total"] = total
            self.all_data.append(run_data)
        #self.all_data.sort(key=lambda x: x["score"])


    def discounted_rewards(self, score):
        r = [0]*21
        r[-1] = score
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(xrange(0, len(r))):
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        discounted_r -= np.mean(discounted_r)
        discounted_r /= np.std(discounted_r)
        return discounted_r

    def get_run_data(self):
        #TODO: for now, we are using just the first 14 features
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_probs = []
        USE_ALL = True
        for r in self.all_data:
            _x = []
            for run in r["episode_data"]:
                trace = []
                states = []
                actions = []
                probs = []
                if USE_ALL:
                    for callsite in run:
                        state = []
                        state.extend(callsite[0][:14])
                        trace.append(callsite[1])
                        full_trace = []
                        full_trace.extend(trace)
                        full_trace.extend([0]*(21-len(full_trace)))
                        state.extend(full_trace)
                        states.append(state)
                        actions.append(callsite[1])
                        probs.append(callsite[2])
                    batch_states.extend(states)
                    batch_actions.extend(actions)
                    batch_rewards.extend(self.discounted_rewards(r["score"]))
                    batch_probs.extend(probs)
        return batch_states, batch_actions, batch_rewards, batch_probs 


    def split_dataset(self, test_size=0.33):
        #TODO: for now, we are using just the first 14 features
        self.X = []
        self.Y = []
        USE_ALL = True
        for r in self.all_data:
            _x = []
            for run in r["episode_data"]:
                trace = []
                if USE_ALL:
                    for callsite in run:
                        features = []
                        features.extend(callsite[0][:14])
                        trace.append(callsite[1])
                        full_trace = []
                        full_trace.extend(trace)
                        full_trace.extend([0]*(21-len(full_trace)))
                        features.extend(full_trace)
                        _x.append(features)
                else:
                    for callsite in run:
                        if callsite[1]==0:
                            continue
                        else:
                            features = []
                            features.extend(callsite[0])
                            _x.append(features[:14])
            seq_len = len(_x)
            self.X.append(torch.FloatTensor(_x).view(seq_len, 1, -1))
            self.Y.append(torch.FloatTensor([r["score"]]).view(1, 1, 1))
            print(len(_x), r["score"])
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size = test_size, random_state=42)
        return self.X_train, self.X_test, self.Y_train, self.Y_test

    def dump(self):
        for r in self.all_data:
            print("score:", r["score"])
            print("number of call sites:", r["total"])
            print("number of passes:", len(r["episode_data"]))
            for run in r["episode_data"]:
                for callsite in run:
                    print(callsite)
                print("------")
        print("best score", self.all_data[0]["score"])
        print("worst score", self.all_data[-1]["score"])
        print("discounted r all_data[0]", self.all_data[0]["discounted_r"])
if __name__== "__main__":
    OCCAM_HOME = os.environ['OCCAM_HOME']
    datapath = os.path.join(OCCAM_HOME, "examples/portfolio/tree/slash") 
    dataset = Dataset(datapath, n_unused_stat = 3, size = 4)
    dataset.dump()
