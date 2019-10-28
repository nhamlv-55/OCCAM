import gym
from gym import error, spaces, utils
from gym.utils import seeding
import subprocess
import os
class OccamGymEnv(gym.Env):
    def __init__(self, workdir, mode, idx):
        self.idx = idx
        self.counter = 0
        self.workdir = workdir
        self.mode = mode
        self.reset()
    def step(self, action):
        with grpc.insecure_channel('localhost:50051') as channel:
            stub = Previrt_pb2_grpc.QueryOracleStub(channel)
            response = stub.Step(action)
            obs = response.obs
            reward = response.reward
            done = response.done
        return obs, reward, done, info
    def reset(self):
        subprocess.Popen("python Connector.py > log_connector".split())
        print("server is up")
        occam_command = "./build.sh --devirt none -g -epsilon %s -folder %s 2>/dev/null"%("-1", self.idx)
        subprocess.check_output(occam_command.split(), cwd = self.workdir)
    def render(self):
        pass
    def close(self):
        pass
