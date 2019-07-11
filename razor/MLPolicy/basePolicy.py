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
        self.get_meta()
        self.memory = ReplayMemory(100000)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trace_len = 40 #a raw estimate of total number of steps in 1 episode. only use to decay epsilon

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
        print("running a trial")
        output = self.net.forward(torch.tensor(self.metadata["sample_inputs"]).view(1, -1))
        print("trial run's output:", output)
        torch.save(self.net, model_path)
        print("tracing the net...")
        print("sample_inputs:")
        print(self.sample_inputs)
        traced_script_module = torch.jit.trace(self.net, self.sample_inputs)
        print("saving the net...")
        traced_script_module.save(os.path.join(OCCAM_HOME, "model.pt"))

    def train(self):
        pass

    def evaluate(self, tag="eval"):
        _ = subprocess.check_output(("%s -epsilon 0 -folder %s 2>%s.log"%(self.run_command, tag, tag)).split(), cwd = self.workdir)
