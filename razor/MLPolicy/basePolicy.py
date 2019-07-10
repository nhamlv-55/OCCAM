from __future__ import print_function
import torch
import os
import subprocess
from utils import *
OCCAM_HOME = os.environ['OCCAM_HOME']

class BasePolicy(object):
    def __init__(self, workdir, model_path, network_type, network_hp):
        self.run_command = "./build.sh "
        self.workdir = workdir
        self.model_path = model_path
        self.dataset_path = os.path.join(workdir, "slash")
        self.network_type = network_type
        self.network_hp = network_hp
        self.bootstrap_runs = 100
        self.bootstrap(model_path, check_format = True)
        self.memory = ReplayMemory(10000)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trace_len = 40 #a raw estimate of total number of steps in 1 episode. only use to decay epsilon
    def bootstrap(self, model_path, check_format):
        # run slash without the model to see how many features we are using
        if check_format:
            print("run check_format to see if we change the number of features")
            #clear previous runs
            if os.path.exists(self.dataset_path):
                clear_prev_runs = subprocess.check_output(("rm -r %s"%self.dataset_path).split())
            #build 1 2 3 4 ... k
            job_ids = ""
            for jid in range(self.bootstrap_runs):
                job_ids +=" %s"%str(jid)
            #run the jobs
            runners_cmd = "parallel %s -epsilon 10 -folder {} 2>/dev/null  ::: %s"%(self.run_command, job_ids)
            print(runners_cmd)
            runners = subprocess.check_output(runners_cmd.split(), cwd = self.workdir)
        dataset_bootstrap = Dataset(self.dataset_path)
        dataset_bootstrap.sort()
        dataset_bootstrap.dump()
        print("features_len:", dataset_bootstrap.features_len)
        print("mean:", dataset_bootstrap.mean)
        print("std:", dataset_bootstrap.std)
        self.sample_inputs = torch.zeros(1, dataset_bootstrap.features_len)
        if self.network_hp is not None:
            self.net = self.network_type(dataset_bootstrap, network_hp)
        else:
            self.net = self.network_type(dataset_bootstrap)
        self.dataset_bootstrap = dataset_bootstrap
        print(self.net)
        trial = self.net(self.sample_inputs)
        print("trial:", trial)
        self.save_model(model_path)

    def load(self, model_path):
        self.net = torch.load(model_path)

    def save_model(self, model_path):
        torch.save(self.net, model_path)
        traced_script_module = torch.jit.trace(self.net, self.sample_inputs)
        traced_script_module.save(os.path.join(OCCAM_HOME, "model.pt"))

    def train(self):
        pass

    def evaluate(self, tag="eval"):
        _ = subprocess.check_output(("%s -epsilon 0 -folder %s"%(self.run_command, tag)).split(), cwd = self.workdir)
