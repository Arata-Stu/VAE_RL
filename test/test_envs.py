import sys 
sys.path.append('..')

import gymnasium as gym

from envs.car_racing import CarRacingWithInfoWrapper

env = gym.make('CarRacing-v3')
env = CarRacingWithInfoWrapper(env, width=64, height=64)

obs, info = env.reset()
print("obs keys", obs.keys())
print("observation space", env.observation_space)
print("action space", env.action_space)
