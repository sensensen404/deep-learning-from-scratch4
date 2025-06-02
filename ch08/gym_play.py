import numpy as np
import gym

env = gym.make('CartPole-v1', render_mode="human")
state = env.reset()
done = False

while True:
    env.render()
    action = np.random.choice([0, 1])
    next_state, reward, done, truncated, info = env.step(action)
env.close()