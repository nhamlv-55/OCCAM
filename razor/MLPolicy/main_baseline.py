import os
from OccamGym import OccamGymEnv
from Connector import Mode
OCCAM_HOME = os.environ["OCCAM_HOME"]
env = OccamGymEnv(workdir = os.path.join(OCCAM_HOME, "examples/portfolio/tree"),
                  mode = Mode.TRAINING,
                  idx = "2",
                  metric = "ROP gadgets",
                  connection = 'localhost:50051')
obs = env.reset()
while True:
    action = True
    obs, rewards, dones, info = env.step(action)
    if dones:
        break
