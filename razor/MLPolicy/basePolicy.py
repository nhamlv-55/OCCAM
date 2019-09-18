from __future__ import print_function
import torch
import os
import subprocess
from utils import *
from grpc_server import QueryOracleServicer
OCCAM_HOME = os.environ['OCCAM_HOME']

class BasePolicy(object):
    def __init__(self, workdir, model_path, network_type, network_hp, grpc_mode, debug):
        self.run_command = "./build.sh --devirt none"
        self.workdir = workdir
        self.model_path = model_path
        self.dataset_path = os.path.join(workdir, "slash")
        self.network_type = network_type
        self.network_hp = network_hp
        self.get_meta()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trace_len = 40 #a raw estimate of total number of steps in 1 episode. only use to decay epsilon
        self.grpc_mode = grpc_mode
        if self.grpc_mode:
            self.run_command +=" -g "
        self.debug = debug
    def get_meta(self):
        with open(os.path.join(self.workdir, "metadata.json")) as json_file:  
            self.metadata = json.load(json_file)
            self.sample_inputs = torch.tensor(self.metadata["sample_inputs"]).view(1, -1)
        print("init the policy with :")
        for k in self.metadata:
            print(k, self.metadata[k])

    def load(self, model_path):
        self.net = torch.load(model_path)

    def save_model(self, model_path):
        if self.debug: print("running a trial")
        if self.net.net_type=="UberNet":
            output = self.net.forward(torch.tensor(np.ones([1, 2+2*self.metadata["max_sequence_len"]])))
        else:
            output = self.net.forward(torch.tensor(self.metadata["sample_inputs"]).view(1, -1))
        if self.debug: print("trial run's output:", output)
        torch.save(self.net, model_path)
        if self.debug: print("tracing the net...")
        if self.debug: print("sample_inputs:")
        if self.debug: print(self.sample_inputs)
        if not self.grpc_mode:
            traced_script_module = torch.jit.trace(self.net, self.sample_inputs)
            print("saving the net...")
            traced_script_module.save(os.path.join(OCCAM_HOME, "model.pt"))

    def run_policy(self, no_of_sampling, eps_threshold, iteration):
        #clear previous runs
        if os.path.exists(self.dataset_path):

            clear_prev_runs = subprocess.check_output(("mv %s %s_run%s"%(self.dataset_path, self.dataset_path, str(iteration)) ).split())
        job_ids = ""
        for jid in range(no_of_sampling):
            job_ids +=" %s"%str(jid)
        runners_cmd = "parallel %s -epsilon %s -folder {} 2>/dev/null  ::: %s"%(self.run_command, eps_threshold, job_ids)
        if self.debug: print(runners_cmd)
        if self.debug: print("workdir:", self.workdir)
        runners = subprocess.check_output(runners_cmd.split(), cwd = self.workdir)
        return runners
    
    def train(self):
        pass

    def forward(self, input):
        raise NotImplementedError()
    def evaluate(self, tag="eval"):
        _ = subprocess.check_output(("%s -epsilon 0 -folder %s 2>%s.log"%(self.run_command, tag, tag)).split(), cwd = self.workdir)
