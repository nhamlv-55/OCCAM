import os
from OccamGym import OccamGymEnv
from Connector import Mode
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

OCCAM_HOME = os.environ["OCCAM_HOME"]
env = OccamGymEnv(workdir = os.path.join(OCCAM_HOME, "examples/portfolio/tree"),
                  mode = Mode.TRAINING,
                  idx = "2",
                  metric = "ROP gadgets",
                  connection = 'localhost:50051')
model = DQN(MlpPolicy, env, verbose=1, batch_size = 1, learning_starts = 2, target_network_update_freq = 5 )
model.learn(total_timesteps=25000)
model.save("occam")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    print(action)
    obs, rewards, dones, info = env.step(action)
    env.render()
