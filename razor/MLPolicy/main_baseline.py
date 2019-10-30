import os
import logging

from tensorflow.python.util import  deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

from OccamGym import OccamGymEnv
from Connector import Mode
OCCAM_HOME = os.environ["OCCAM_HOME"]
#DQN
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines.deepq.policies import MlpPolicy
# from stable_baselines import DQN

# env = OccamGymEnv(workdir = os.path.join(OCCAM_HOME, "examples/portfolio/tree"),
#                   mode = Mode.TRAINING,
#                   idx = "1",
#                   metric = "ROP gadgets",
#                   connection = 'localhost:50001')
# model = DQN(MlpPolicy, env, verbose=1, batch_size = 1, learning_starts = 2, target_network_update_freq = 5 )
# model.learn(total_timesteps=25000)
# model.save("occam")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     print(action)
#     obs, rewards, dones, info = env.step(action)
#     env.render()



#A2C
# import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import A2C

def make_env(idx):
    return OccamGymEnv(workdir = os.path.join(OCCAM_HOME, "examples/portfolio/tree"),
                    mode = Mode.TRAINING,
                    idx = str(idx),
                    metric = "ROP gadgets",
                    connection = 'localhost:%s'%str(50000+idx))
 

def main():
    # multiprocess environment
    n_cpu = 2
    env_list = []
    for i in range(n_cpu):
        a_env = OccamGymEnv(workdir = os.path.join(OCCAM_HOME, "examples/portfolio/tree"),
                        mode = Mode.TRAINING,
                        idx = str(i),
                        metric = "ROP gadgets",
                        connection = 'localhost:%s'%str(50000+i))
        env_list.append(a_env)

    #print([lambda: make_env(i) for i in range(n_cpu)])
    #return
    env = SubprocVecEnv(env_list)
    model = A2C(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("a2c_cartpole")

    del model # remove to demonstrate saving and loading

    model = A2C.load("a2c_cartpole")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()


# import gym

# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.vec_env import SubprocVecEnv
# from stable_baselines import A2C

# def main():
#     # multiprocess environment
#     n_cpu = 4
#     env = SubprocVecEnv([lambda: gym.make('CartPole-v1') for i in range(n_cpu)])

#     model = A2C(MlpPolicy, env, verbose=1)
#     model.learn(total_timesteps=25000)
#     model.save("a2c_cartpole")

#     del model # remove to demonstrate saving and loading

#     model = A2C.load("a2c_cartpole")

#     obs = env.reset()
#     while True:
#         action, _states = model.predict(obs)
#         obs, rewards, dones, info = env.step(action)
#         env.render()


if __name__=="__main__":
    main()
