#####
#
# @author: Felipe Iturralde
#
######

import torch
import torch.nn.functional as F

# import sys
import os
import random
import numpy as np
import pandas as pd
from datetime import datetime as dt
from collections import deque

from unityagents import UnityEnvironment

# env_path = os.path.abspath("../Banana.app")
env_path = "Banana.app"

class Train_Agent():
  """
  Trains a DRL agent designed to learn the challenge of project 1
  
  Args:
  - agent (str): The name of one of the available agent implementations
  """
  
  def __init__(self, agent_implementation):
    
    self.agent_implementation = agent_implementation
    self.dqn_agent = self._set_agent_import_ref()
    
    # dataframe to score the training progress time series
    # self.training_progress = pd.DataFrame(column = ['agent', 'session', 'score', 'epsilon', 'beta'])
    
    return
  
  def _init_env(self):
    """
    Initializes a new Banana environment
    """
    
    try:
      env = env = UnityEnvironment(file_name = env_path)
    except Exception as err:
      msg = "Unable to launch env @ {}".format(env_path)
      raise RuntimeError(msg) from err
    
    return env
  
  def _set_agent_import_ref(self):
    """
    Returns an Agent instance
    """
    
    if self.agent_implementation == 'dqn_agent_1_2':
      from agent import dqn_agent_1_2 as dqn_agent
    elif self.agent_implementation == 'dqn_agent_1_2_3':
      from agent import dqn_agent_1_2_3 as dqn_agent
    elif self.agent_implementation == 'dqn_agent_1_3_4':
      from agent import dqn_agent_1_3_4 as dqn_agent
    else: 
      msg = "'{}' is not a registered Agent".format(self.agent_implementation)
      raise RuntimeError(msg)
    
    return dqn_agent
  
  # methods
  
  def Train(self, n_episodes = 2500, max_t = 1000, target_score = np.inf, 
            eps_start = 1.0, eps_end = 0.01, eps_decay = 0.995, 
            priority_replay = False, exp_replay_alpha = 0.2):
    """
    Trains the agent, and records training performance
    
    The agent is trained for n_episodes or untill it reaches target_score.
    Each episode lasts max_t interactions.
    
    Args:
    - n_episodes (int) default 2500: number of episodes to train the agent for
    
    - max_t (int) default 1000: number of t (interactions) in an episode
    
    - target_score (float) default inf: training stops if the agent's 
      training mean score, over the last 100 episodes, reaches target_score
      
    - eps_start (float) default 1.0: epsilon-greedy hyperparameter initial setting
    
    - eps_end (float) defaul 0.01: epsilon-greedy hyperparameter end setting
    
    - eps_decay (float) default = 0.995: rate of decay of epsilon
    
    - priority_replay (bool) default False: the agent implements a priority experience replay buffer
    
    - exp_replay_alpha (float) default 0.2: Value of the priority tuning parameter alpha
      for the experience selection priority
      
    Returns:
    - agent (object): Instance of agent_implementation
    - training_data (DataFrame): contains details of the training session
    """
    
    # initialize and configure environment and agent
    
    env = self._init_env()
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=True)[brain_name]

    state_size = len(env_info.vector_observations[0])
    action_size = env.brains[brain_name].vector_action_space_size
    seed = 0
    
    agent = self.dqn_agent.Agent(state_size, action_size, seed, 
                                 **{'exp_replay_alpha': exp_replay_alpha}
                                )
    
    # initialize and configure training context
    
    scores = []                          # episode score list
    scores_window = deque(maxlen = 100)  # episode's last 100 scores
    
    eps = eps_start
    beta = 0
    
    # training
    
    for i_episode in range(1, n_episodes + 1):
      
      env_info = env.reset(train_mode = True)[brain_name]
      state = env_info.vector_observations[0]
      score = 0
      
      for t in range(max_t):
        
        action = agent.act(state, eps)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        
        # this env is never done == True, but we check anyway, 
        # just in case "surprise!"
        if done:
          break
      
      # process progress
      
      scores_window.append(score)
      scores.append([i_episode, score, eps, beta])
      
      print('\rEpisode: {}\tAverage Score: {:.2f}\teps: {:.4f}\tbeta: {:.6f}'
          .format(i_episode, np.mean(scores_window), eps, beta), end="")
    
      timestamp = dt.now().strftime('%Y-%m-%d %H:%M:%S')
      if i_episode % 100 == 0:
        print('\r{} - Episode: {}\tAverage Score: {:.2f}\teps: {:.4f}\tbeta: {:.6f}'
          .format(timestamp, i_episode, np.mean(scores_window), eps, beta))

      if np.mean(scores_window) >= target_score:
        print('\n{} - Environment solved in:'.format(timestamp))
        print('Episode: {}\tAverage Score: {:.2f}\teps: {:.4f}\tbeta: {:.6f}'
          .format(i_episode, np.mean(scores_window), eps, beta))

        break
        
      # set up next round
      
      eps = max(eps_end, eps_decay * eps)
      
      if priority_replay:
        beta = agent.update_importance_w_beta(max_t * i_episode)
    
    # close training housekeeping
    
    training_data = pd.DataFrame(data = scores, 
                                 columns = ['episode', 'score', 'epsilon', 'beta']
                                )
    env.close()
    agent.finalize()
    
    return agent, training_data
  
  
      
    
    
    
    
  
  