#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import gym
import random


# In[2]:


# Create environment using OpenAI gym
env = gym.make("Taxi-v3")
action_size = env.action_space.n #possible actions for taxi
state_size = env.observation_space.n #possible states to encounter for taxi
print("Action space size: ", action_size)
print("State space size: ", state_size)


# In[3]:


# Initialize Q-learning table to 0
Q = np.zeros((state_size, action_size))


# In[4]:


# Hyperparamters
episodes = 2000         # Total training episodes
time_steps = 100                 # time-steps per episode
alpha = 0.7                      # Learning rate
gamma = 0.618                    # Discounting rate


# In[5]:


# Exploration parameters
epsilon = 1                   # Exploration rate
prob_epsilon = 1               # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.01             # Decay rate for exploration prob


# In[6]:


# Learning Phase
rewards = []   # list of rewards

for episode in range(episodes):
    state = env.reset()    # Reset the environment
    total_rewards = 0
    
    for step in range(time_steps):
        # Choose an action (a) among the possible states (s)
        exp_exp_tradeoff = random.uniform(0, 1)   # choose a random number
        
        # If this number > epsilon, select the action corresponding to the biggest Q value for this state (Exploitation)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(Q[state,:])        
        # Else choose a random action (Exploration)
        else:
            action = env.action_space.sample()
        
        # Observe the outcome state(s') and reward (r)
        next_state, reward, done, info = env.step(action)

        # Update the Q table using the Reinforcement learning equation
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action]) 
        total_rewards += reward  # increase the total reward        
        state = next_state         # Update the state
        
        # If we reach the end of the episode
        if done == True:
            print ("Total rewards for episode {}: {}".format(episode, total_rewards))
            break
    
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (prob_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    
    # append the episode cumulative reward to the list
    rewards.append(total_rewards)

print ("Average rewards over time: " + str(sum(rewards)/episodes)) 


# In[7]:



x = range(episodes)
plt.plot(x, rewards)
plt.xlabel('episode')
plt.ylabel('Total rewards')
plt.show()


# In[ ]:




