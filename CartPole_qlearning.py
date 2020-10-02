#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import required libraries
import gym
import numpy as np
import math
from collections import deque


# In[2]:


#create a class for various cartpole functions
class CartPole():
    def __init__(self, buckets=(1, 1, 6, 12,), n_episodes=1000, n_win_episode=195, min_alpha=0.1, min_epsilon=0.1, gamma=1.0, ada_divisor=25, max_env_steps=None, monitor=False):  #declare hyperparameters and set episodes to 1000
        self.buckets = buckets # discrete range for feature space
        self.n_episodes = n_episodes # number of episodes
        self.n_win_episode = n_win_episode # average episodes required to win
        self.min_alpha = min_alpha # learning rate
        self.min_epsilon = min_epsilon # exploration rate
        self.gamma = gamma # discount rate
        self.ada_divisor = ada_divisor # only for development purposes

        self.env = gym.make('CartPole-v0')       #create cart-pole environment
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
        if monitor: self.env = gym.wrappers.Monitor(self.env, 'tmp/cartpole-1', force=True)

        # Initialise Q-table with zero
        self.Q = np.zeros(self.buckets + (self.env.action_space.n,))

    # Discretizing function for Q table
    def discretize(self, obs):
        upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50)]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    # Choose action based on q-learning 
    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.Q[state])

    # Update Q-value based on reinforcement learning equation
    def update_q(self, state_old, action, reward, state_new, alpha):
        self.Q[state_old][action] += alpha * (reward + self.gamma * np.max(self.Q[state_new]) - self.Q[state_old][action])

    # Adaptive Exploration Rate
    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1, 1.0 - math.log10((t + 1) / self.ada_divisor)))

    # Adaptive Learning Rate
    def get_alpha(self, t):
        return max(self.min_alpha, min(1.0, 1.0 - math.log10((t + 1) / self.ada_divisor)))

    #q learning algorithm
    def run(self):

        for e in range(self.n_episodes):
            # Discretize states 
            current_state = self.discretize(self.env.reset())

            # alpha and epsilon decayed over time
            alpha = self.get_alpha(e)
            epsilon = self.get_epsilon(e)
            done = False
            i = 0

            while not done:
                # Render environment
                self.env.render()

                # Choose action according to q-learning policy
                action = self.choose_action(current_state, epsilon)
                obs, reward, done, _ = self.env.step(action)
                new_state = self.discretize(obs)

                # Update Q-Table
                self.update_q(current_state, action, reward, new_state, alpha)
                current_state = new_state
                i += 1


# In[ ]:


if __name__ == "__main__":

    # Make and run an instance of CartPole class 
    solver = CartPole()
    solver.run()


# In[ ]:




