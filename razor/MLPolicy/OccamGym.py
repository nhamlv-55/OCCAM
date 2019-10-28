import gym
from gym import error, spaces, utils
from gym.utils import seeding
import subprocess
import os
import time
import Previrt_pb2
import Previrt_pb2_grpc
import grpc
class OccamGymEnv(gym.Env):
    def __init__(self, workdir, mode, idx, connection):
        self.idx = idx
        self.counter = 0
        self.workdir = workdir
        self.mode = mode
        self.connection = connection
        self.reset()
    def _get_obs(self):
        return self.step(action = None)
    def step(self, action, q_yes = -1, q_no = -1, state_encoded = "EMPTY"):
        if action is not None:
            prediction =  Previrt_pb2.Prediction(q_no = q_no, q_yes = q_yes, state_encoded = state_encoded, pred = action)
        else:
            prediction =  Previrt_pb2.Prediction(q_no = -99, q_yes = -99, state_encoded = state_encoded, pred = False)
        with grpc.insecure_channel(self.connection) as channel:
            stub = Previrt_pb2_grpc.QueryOracleStub(channel)
            response = stub.Step(prediction)
            obs = response.obs
            reward = response.reward
            done = response.done
            info = response.info
        return obs, reward, done, info
    def reset(self):
        subprocess.Popen("python Connector.py > log_connector".split())
        occam_command = "./build.sh --devirt none -g -epsilon %s -folder %s 2>/dev/null"%("-1", self.idx)
        subprocess.Popen(occam_command.split(), cwd = self.workdir)
        #keep querying until the 1st obs is returned
        time.sleep(2)
        while True:
            try:
                time.sleep(1)
                obs, reward, done, info = self._get_obs()
                if obs[0] != -1:
                    print("env is reset. Got 1st state")
                    return obs, reward, done, info
            except Exception as e:
                print(e)
    def render(self):
        pass
    def close(self):
        pass
