from __future__ import print_function
import os
import glob
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import random
from collections import namedtuple
import json
import subprocess
import argparse
import math
import GSA_util.GSA as gsa

from inst2vec import inst2vec_preprocess as i2v_prep

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
Step   = namedtuple('Step', ('state', 'rnn_state', 'prob', 'action'))
np.set_printoptions(precision=6, suppress=True)
GAMMA = 0.99
DEBUG = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#Plot the rewards
def plot(no_of_sampling, no_of_run, workdir, graph_file, agg, opt,  metric = "Total unique gadgets", ext = "pdf"):
    plt.clf()
    iters = []
    means = []
    stdevs = []

    def read_profiler(iteration, run, metric):
        res_path = os.path.join(workdir, "slash_run%s/run%s/0AfterSpecialization_results.txt"%(str(iteration), str(run)))
        res = open(res_path, "r").readlines()
        for l in res:
            if metric in l:
                tokens = l.strip().split()
                return int(tokens[0])

    #legacy code
    def read_rop_profiler(iteration, run, metric):
        res_path = os.path.join(workdir, "slash_run%s/run%s/rop_stats.txt"%(str(iteration), str(run)))
        res = open(res_path, "r").readlines()
        rop_count = int(res[-1].strip().split()[-1])
        return rop_count


    def read_rop_stats(iteration, run, metric):
        res_path = os.path.join(workdir, "slash_run%s/run%s/"%(str(iteration), str(run)))
        with open(res_path+"/gadget_count.json", "r") as f:
            data = json.load(f)
            return int(data[metric])

    for i in range(1, no_of_run):
        run_results = []
        for j in range(no_of_sampling):
            if "gadgets" in metric:
                run_results.append(read_rop_stats(i,j, metric))
            elif "legacy" in metric:
                run_results.append(read_rop_profiler(i, j, metric))
            else:
                run_results.append(read_profiler(i, j, metric))
        iters.append(i)
        means.append(np.mean(run_results))
        stdevs.append(np.std(run_results))

    plot_name = "%s_%s_%s.%s"%(graph_file, no_of_sampling, no_of_run, ext)
    json_name = "%s_%s_%s.%s"%(graph_file, no_of_sampling, no_of_run, "json")

    x = np.array(iters)
    y = np.array(means)
    e = np.array(stdevs)

    plt.errorbar(x, y, e, linestyle='None', marker='^')
    plt.axhline(y=agg, color='r', linestyle='-')
    plt.axhline(y=opt, color='orange', linestyle='-')
    plt.savefig(os.path.join(workdir, plot_name))

    #save the summary
    summary = {"iters": iters, "means": means, "stdevs": stdevs, "agg": agg, "opt": opt}
    with open(os.path.join(workdir, json_name), "w") as f:
        json.dump(summary, f, indent=4, sort_keys=True)


#ReplayMemory
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
    def __init__(self, folder, metric, collect_encoded_state = False, n_unused_stat = 3, size=99999999):
        self.folder = folder
        self.metric = metric
        print(self.folder, self.metric)
        print("collect_encoded_state=", collect_encoded_state)
        self.n_unused_stat = n_unused_stat
        self.all_data = []
        self.collect_encoded_state = collect_encoded_state
        self.collect(size)
        if self.no_good_runs > 0:
            self.calculate_stats()
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
            sub_episode = []
            #read .csv file
            with open(fname, "r") as f:
                f_data = f.readlines()
            with open(fname+".state_encoded") as f_encoded:
                f_enc_data = f_encoded.readlines()

            #broken run
            if self.collect_encoded_state and len(f_data)!=len(f_enc_data):
                if DEBUG:
                    print("error in %s"%fname)
                    print("there are only %s lines in .encoded_state"%str(len(f_enc_data)))
                return

            #read data
            for i in range(len(f_data)):
                l = f_data[i]
                if self.collect_encoded_state and len(f_enc_data)>0:
                    rnn_state = f_enc_data[i]
                    rnn_state = [int(t) for t in rnn_state.strip().split()]
                    #print(len(rnn_state))
                    #print(rnn_state[:10])
                else:
                    rnn_state = []
                total+=1
                tokens = l.strip().split(',')
                tokens = [float(t) for t in tokens]
                raw_data.append(tokens)
                state = tokens[:-self.n_unused_stat]
                prob = tokens[-3:-1]
                action = tokens[-1]
                sub_episode.append(Step(state, rnn_state, prob, action))
            if len(sub_episode)>0:
                episode_data.append(sub_episode)
        return raw_data, episode_data, total

    def get_stat(self, run):
        result = {}
        #get rop gadgets count
        with open(run+"/gadget_count.json", "r") as f:
            data = json.load(f)
            for k in data:
                result[k]= data[k]
 
        #get instruction counts
        with open(run+"/0AfterSpecialization_results.txt", "r") as f:
            for l in f.readlines():
                if "[" in l:
                    continue
                tokens = l.strip().split()
                v = int(tokens[0])
                k = " ".join(tokens[1:])
                result[k] = v
        if "agg" in run:
            self.agg_results = result
        if "none" in run:
            self.none_results = result
        return result
    
    def score(self, result):
        return result[self.metric]*1.0
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

    def calculate_stats(self):
        raw_data_np = np.array(self.raw_data)
        print(len(raw_data_np))
        self.mean = np.mean(raw_data_np, 0)
        self.std  = np.std(raw_data_np, 0)
        self.maxx = np.max(raw_data_np, 0)
        self.minn = np.min(raw_data_np, 0)
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
        self.no_good_runs = 0
        for r in runs[:size]:
            run_data = {}
            #print(r)
            csv_files = glob.glob(r+"/*.csv")
            csv_datas = self.merge_csv(csv_files)
            #only use data from unbroken runs
            if csv_datas is not None:
                _, episode_data, total = csv_datas
            else:
                continue
            result = self.get_stat(r)
            self.raw_data.extend(_)
            run_data["episode_data"] = episode_data
            run_data["score"] = self.score(result)
            run_data["raw_result"] = result
            run_data["total"] = total
            self.all_data.append(run_data)
            self.no_good_runs+=1
        print("collected %s good runs out of %s runs"%(str(self.no_good_runs), str(size)))

    def sort(self):
        self.all_data.sort(key=lambda x: x["score"])

    # for Policy Gradient
    def get_trajectory_data(self, normalize_rewards = False):
        batch_states = []
        batch_rnn_states = []
        batch_actions = []
        batch_rewards = []
        batch_probs = []
        for eps in self.all_data:
            rnn_states = []
            states = []
            actions = []
            probs = []
            
            for sub_episode in eps["episode_data"]:
                for step in sub_episode:
                    rnn_states.append(step.rnn_state)
                    states.append(step.state)
                    actions.append(step.action)
                    probs.append(step.prob)
            #try aliased states shuffling (only for 3 )
            if(len(states)==3):
                random.shuffle(states)
            batch_rnn_states.extend(rnn_states)
            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_probs.extend(probs)
            #rewards = [eps["score"]]*len(states)
            rewards = [0]*len(states)
            rewards[-1] = eps["score"]
            batch_rewards.extend(discount_rewards(rewards, GAMMA))
        if normalize_rewards:
            if DEBUG: print("before norm:", batch_rewards)
            batch_rewards -= np.mean(batch_rewards)
            if DEBUG: print("after subtract mean:", batch_rewards)
            batch_rewards /= np.std(batch_rewards)
            if DEBUG: print("after / std : ", batch_rewards)
        return batch_states, batch_rnn_states, batch_actions, batch_rewards, batch_probs 
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

def discount_rewards(rewards, gamma):
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for i in reversed(range(0, len(rewards))):
        cumulative_rewards = cumulative_rewards * gamma + rewards[i]
        discounted_rewards[i] = cumulative_rewards
    return discounted_rewards


def gen_new_meta(workdir, bootstrap_runs, run_command, get_rop_detail = False, metric = None):
    if metric is None:
        print("Need to provide metric")
        return
    metadata = {}
    metadata["padding_idx"] = 8565 #hard coded. = no of entry in inst2vec vocab
    metadata["max_sequence_len"] = 2000 #hardcoded. assuming the longest len of a function is 2000
    metadata["max_args_len"] = 100 #hardcoded. assuming the longest len of args is 100
    binary_name = workdir.split("/")[-1]
    print("running on binary file %s.bc"%binary_name)
    # grab the struct dictionaries from the bc
    if not os.path.exists(os.path.join(workdir, binary_name+".ll")):
        #run llvm-dis
        bc_path = os.path.join(workdir, binary_name+".bc")
        ll_path = os.path.join(workdir, binary_name+".ll")
        llvm_dis_cmd = "llvm-dis %s -o=%s"%(bc_path, ll_path)
        print("running %s"%llvm_dis_cmd)
        subprocess.check_output(llvm_dis_cmd.split(), cwd = workdir)

    with open(os.path.join(workdir, binary_name+".ll"), "r") as ll_file:
        raw_data = ll_file.read().splitlines()
        _, struct_dict = i2v_prep.construct_struct_types_dictionary_for_file(raw_data)
    
    metadata["struct_dict"] = struct_dict    
    dataset_path = os.path.join(workdir, "slash")
    # run slash without the model to see how many features we are using
    print("run check_format to see if we change the number of features")
    print("metric:", metric)
    #clear previous runs
    if os.path.exists(dataset_path):
        print("clearning previous runs...")
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

    #use GSA to get ROP stats
    if get_rop_detail:
        print("getting ROP details...")
        original = os.path.join(workdir, "slash/runnone/%s"%binary_name)
        variants_dict = {"agg": os.path.join(workdir, "slash/runagg/%s"%binary_name)}
        for i in range(bootstrap_runs):
            variants_dict[str(i)]=os.path.join(workdir, "slash/run%s/%s"%(str(i), binary_name))
        directory_name = os.path.join(workdir, "gsa_stat")
        if not os.path.exists(directory_name):
            os.mkdir(directory_name)
        gsa.run_gsa(original, variants_dict, directory_name)
    dataset_bootstrap = Dataset(dataset_path, metric)
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
    metadata["max_score"] = dataset_bootstrap.all_data[0]["score"]
    metadata["min_score"] = dataset_bootstrap.all_data[-1]["score"]
    metadata["maxx"] = dataset_bootstrap.maxx.tolist()
    metadata["minn"] = dataset_bootstrap.minn.tolist()
    metadata["metric"] = dataset_bootstrap.metric
    metadata["agg_results"] = dataset_bootstrap.agg_results
    metadata["none_results"]= dataset_bootstrap.none_results
    with open(os.path.join(workdir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4, sort_keys=True)

class Atomizer(object):
    '''
    encode the llvm IR (the token-format)
    In this encoding scheme, every instruction is a list of `words`
    Each `words` is a list of `chars` (`symbols`)
    The first token (opt code) in each instruction is regarded as 1 `char`
    For each operand in the instruction, we split them into chunk of continuous char and digits,
    for examples: `br tobool1  if.then103 if.else103`
    => [['br'],
    ['tobool', '1'],
    ['if','then', '1','0','3'],
    ['if','else', '1','0','3']
    ]
    We also split at `.` (dot) because this is a special character
    
    '''
    def __init__(self):
        self.symbol_vocab = set()
        self.symbol2idx = {}
        self.idx2symbol = {}
        for i in range(10):
            self.symbol2idx[str(i)] = i
            self.idx2symbol[i] = str(i)
        self.word2idx = {}
        self.idx2word = {}

    def handle_symbol(self, sym):
        if sym not in self.symbol2idx:
            self.symbol2idx[sym] = len(self.symbol2idx)
            self.idx2symbol[len(self.idx2symbol)] = sym

    
    def split_token(self, token):
        '''
        split a token into a list of symbols and add these symbols to the symbol vocab
        '''
        token+="0" #a hack to not have to handle edge cases
        symbols = [] 
        current_s = ""
        for c in token:
            if c==".":
                self.handle_symbol(current_s)
                symbols.append(self.symbol2idx[current_s])
                current_s = ""
            elif c.isdigit():
                self.handle_symbol(current_s)
                symbols.append(self.symbol2idx[current_s])
                current_s = c
            else:
                if current_s.isdigit():
                    symbols.append(self.symbol2idx[current_s])
                    current_s = c
                else: current_s+=c
        return symbols

    def normalize_ptr(self, insts):
        '''
        given a list of insts in the token-form, create a map of ptr to map from ptr0x55c5f7f542a0 to a simpler token like ptr__12
        return the ptr_map and the new list of insts
        '''
        ptr_map = {}
        new_insts = []
        for l in insts:
            new_tokens = []
            tokens = l.split()
            for t in tokens:
                if t.startswith("ptr0x"):
                    if t in ptr_map:
                        t = ptr_map[t]
                    else:
                        ptr_map[t] = "ptr__"+str(len(ptr_map))
                        t = ptr_map[t]
                new_tokens.append(t)
            rewritten_inst = " ".join(new_tokens)
            new_insts.append(rewritten_inst)
        return ptr_map, new_insts

    def encode(self, insts):
        '''
        take in a list of insts in the token-format and return an encoded string.
        We can return a numpy here but using a string we are forcing the same format across the log and the policy
        '''
        encoded_str = ""
        ptr_map, new_insts = self.normalize_ptr(insts)
        for inst in new_insts:
            if len(inst)==0:
                continue
            tokens = inst.split()
            opt_code = tokens[0]
            self.handle_symbol(opt_code)
            encoded_str+=str(self.symbol2idx[opt_code])+" "
            for operand in tokens[1:]:
                token_symbols_idx = self.split_token(operand)
                encoded_str+="-".join([str(idx) for idx in token_symbols_idx])
                encoded_str+=" "
            encoded_str+="\n"
        return encoded_str

    def decode(self, array):
        '''
        take in an encoded string and decode it back to original form
        '''
        for l in array:
            tokens = l.strip().split()
            for tok in tokens:
                syms = tok.split("-")
                for sym in syms:
                    print(self.idx2symbol[int(sym)]),
                print(" ")
            print("\n")

if __name__== "__main__":
    OCCAM_HOME = os.environ['OCCAM_HOME']
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', default=100, help='number of trial runs to get meta data')
    parser.add_argument('-f', default=os.path.join(OCCAM_HOME,"examples/portfolio/tree"), help='work_dir')
    parser.add_argument('-m', default="Total unique gadgets", help='metrics to get')
    args = parser.parse_args()
    bootstrap_runs = int(args.n)
    model_path = os.path.join(OCCAM_HOME, "razor/MLPolicy/model") 
    work_dir   = args.f
    metric = args.m
    gen_new_meta(work_dir, bootstrap_runs, "./build.sh ", metric = metric)
