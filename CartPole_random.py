#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import required libraries
import gym
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#function to simulate one episode
def run_episode(env, parameters):
    observation = env.reset()
    cumulativereward = 0 #total number of rewards set to zero at beginning
    for _ in range(200):  #need to keep pole up for 200 time-steps to solve the problem
        action = 0 if np.matmul(parameters,observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        cumulativereward += reward   #increase rewards after each episode
        if done:
            break
    return cumulativereward          


# In[4]:


#random search algorithm for agent
def learn(submit):
    env = gym.make('CartPole-v0')        #create car-pole environment using OpenAI gym
    if submit:
        env.monitor.start('cartpole-experiments/', force=True)

    counter = 0
    bestparams = None
    bestreward = 0
    for _ in range(10000):               #run the method 1000 times
        counter += 1
        parameters = np.random.rand(4) * 2 - 1
        reward = run_episode(env,parameters)
        if reward > bestreward:
            bestreward = reward
            bestparams = parameters
            if reward == 200:
                break

    if submit:
        for _ in xrange(100):
            run_episode(env,bestparams)
        env.monitor.close()

    return counter

results = []
for _ in range(1000):
    results.append(learn(submit=False))
#plot the graph depicting the amount of episodes it took to reach 200 time-steps
plt.hist(results,50,density=1, facecolor='g', alpha=0.75)
plt.xlabel('Episodes required to reach 200')
plt.ylabel('Frequency')
plt.title('Random Search Histogram')
plt.show()

print((np.sum(results) / 1000.0))  #amount of episodes


# In[ ]:




