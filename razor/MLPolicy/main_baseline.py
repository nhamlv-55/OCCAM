import os
from OccamGym import *
from Connector import Mode
OCCAM_HOME = os.environ["OCCAM_HOME"]
env = OccamGymEnv(os.path.join(OCCAM_HOME, "examples/portfolio/tree"), Mode.TRAINING,  "1")
