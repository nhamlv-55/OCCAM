import gym

env = gym.make('OccamGym')
model = PPO2(MlpPolicy, env, verbose = 1)
model.learn(total_timesteps = 10000)
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    
