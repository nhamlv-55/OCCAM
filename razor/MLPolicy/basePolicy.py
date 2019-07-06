import torch
import os
import subprocess
from utils import ReplayMemory
OCCAM_HOME = os.environ['OCCAM_HOME']

class BasePolicy(object):
    def __init__(self, workdir, model_path, network_type, network_hp):
        self.sample_inputs = torch.tensor([4.1,1.76,6.3,0.0,0.18,0.39,1,7.9,3.98,0.98,0.08,0.64,0.78,1.000000, 0]).view(1, -1)
        self.run_command = "./build.sh "
        self.workdir = workdir
        self.model_path = model_path
        self.dataset_path = os.path.join(workdir, "slash")
        self.network_type = network_type
        self.network_hp = network_hp
        self.bootstrap(model_path)
        self.memory = ReplayMemory(10000)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trace_len = 40 #a raw estimate of total number of steps in 1 episode. only use to decay epsilon

    def bootstrap(self, model_path):
        if self.network_hp is not None:
            self.net = self.network_type(network_hp)
        else:
            self.net = self.network_type()
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
