

import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import dqn_model

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64

class Agent():
  """
  Interacts with and learns from the environment
  
  Params:
  - state_size (int): number of s in S
  - action_size (int): number of a in A
  - seed (int): random seed
  """
  
  def __init__(self, state_size, action_size, seed, exp_replay_alpha = 0.2):
    
    # context
    self._context = dict()
    self.context['buffer_size'] = int(1e5)
    self.context['batch_size'] = 64
    self.context['gamma'] = 0.99
    self.context['tau'] = 1e-3
    self.context['lr'] = 5e-4
    self.context['update_rate'] = 4
    self.context['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.context['state_size'] = state_size
    self.context['action_size'] = action_size
    self.context['seed'] = seed
    self.context['importance_w_beta'] = 0.4
    self.context['exp_replay_alpha'] = exp_replay_alpha
    self.context['beta_frames'] = 1e6
    self.context['error_constant'] = 1e-5
    
    # DQ-network
    self.dqn_local = dqn_model.DQNetwork(state_size, action_size, seed).to(self.context['device'])
    self.dqn_local.apply(self.dqn_local.init_weights)
    
    self.dqn_target = dqn_model.DQNetwork(state_size, action_size, seed).to(self.context['device'])
    self.dqn_target.apply(self.dqn_target.init_weights)
    
    self.optimizer = optim.Adam(self.dqn_local.parameters(), lr=self.context['lr'])
    self.beta = self.context['importance_w_beta']
    
    # Experience Replay Memory
    self.replay_buffer = PriorityReplayBuffer(buffer_size = self.context['buffer_size'], 
                                              batch_size = self.context['batch_size'], 
                                              seed = seed, 
                                              prob_alpha = exp_replay_alpha, 
                                              device = self.context['device'])
    
    # time_step
    self.t_step = 0
    
    return
  
  # properties
  
  @property
  def context(self):
    """
    Learning context variables and hyperparameters
    """
    
    return self._context
  
  # functions
  
  def finalize(self):
    """
    """
    
    self.dqn_local = self.dqn_local.to('cpu')
    self.dqn_target = self.dqn_target.to('cpu')
    
    return
  
  def update_importance_w_beta(self, frames_n):
    """
    """
    beta = ((self.context['importance_w_beta'] + frames_n) * 
            (1.0 - self.context['importance_w_beta']) / 
            (self.context['beta_frames'])
           )
    self.beta = min(1.0, beta)
    
    return self.beta
  
  def step(self, state, action, reward, next_state, done):
    """
    Implements a DQN actions taken at each step
    """
    
    # compute the experience priority
    # e_delta = self.get_observation_error(state, action, reward, next_state)
    # p = e_delta + self.context['error_constant']
    p = self.replay_buffer.priority.max() if len(self.replay_buffer) > 0 else 1.0
    
    # store experience in replay buffer
    self.replay_buffer.insert(state, action, reward, next_state, done, 
                              p = p)
    
    # Learn
    self.t_step = (self.t_step + 1) % self.context['update_rate']
    if self.t_step == 0:
      if len(self.replay_buffer) > self.context['batch_size']:
        experiences = self.replay_buffer.sample(self.beta)
        self.learn(experiences)
      
    return
  
  def get_observation_error(self, state, action, reward, next_state):
    """
    """
    with torch.no_grad():
      s = torch.from_numpy(state).unsqueeze(0).float().to(self.context['device'])
      a = torch.from_numpy(np.array([action])).unsqueeze(0).long().to(self.context['device'])
      sn = torch.from_numpy(next_state).unsqueeze(0).float().to(self.context['device'])

      # get best current w for q(s', A) := max(S', A, w)
      Q_sn_r = self.dqn_local(sn).max(1)[0].detach()

      # get expected (current) w for q(s, a)
      Q_s_r = self.dqn_local(s).gather(1, a).detach()

      # compute the error_delta
      e_delta = reward + (self.context['gamma'] * Q_sn_r)

      # compute the error (loss)
      # l = F.mse_loss(Q_s_r, e_delta)
      l = (e_delta - Q_s_r) ** 2
    
    return l.item()
  
  def learn(self, experiences):
    """
    Learn.
    i.e. update value parameters using experiences
    
    Params:
    - experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
    """
    
    states, actions, rewards, next_states, dones, sample_index, importance_w = experiences
    
    # get expected (current) w for q(s, a)
    Q_current = self.dqn_local(states).gather(1, actions)

    # Using Double Q-learning: 
    ## get best current w for q(s', A) := max(S', A, w)
    Q_current_actions = self.dqn_local(next_states)
    Q_current_next_best_actions = Q_current_actions.detach().max(1)[1].unsqueeze(1)
    Q_current_next_rewards = Q_current_actions.gather(1, Q_current_next_best_actions)
    ## Evaluate
    ## get best target w for q'(S', best_actions, w-)
    Q_target_next_rewards = self.dqn_target(next_states).gather(1, 
                                                                Q_current_next_best_actions
                                                               )

    # compute the Q_targets:  If the episode is done we only keep the reward
    ## with double Q-learning w- dampers the impact of optimistic w
    Q_targets = rewards + (self.context['gamma'] * Q_target_next_rewards * (1 - dones))
    
    # apply importance sampling weight and compute new priority of experienses
    # isw = torch.from_numpy(importance_w).unsqueeze(1).to(self.context['device'])
    
    # compute priorities
    Gt = rewards + (self.context['gamma'] * Q_current_next_rewards * (1 - dones))
    # priorities = (((Gt - Q_current) ** 2) + self.context['error_constant']).squeeze(1).tolist()
    priorities = (torch.abs(Gt - Q_current) + self.context['error_constant']).squeeze(1).tolist()
    
    # apply importance sampling weight and compute the error (loss) for the minibatch
    isw = torch.from_numpy(importance_w).unsqueeze(1).to(self.context['device'])
    loss = F.mse_loss(Q_current * isw, Q_targets * isw)
    
    # minimize the loss
    self.optimizer.zero_grad()
    loss.backward() 
    # loss.mean().backward()
    self.optimizer.step()
    
    # update target network
    self.soft_update(local_model = self.dqn_local, target_model = self.dqn_target)
    
    # update experience priority
    self.replay_buffer.update_priority(priorities, sample_index)
    
    return
  
  def soft_update(self, local_model, target_model):
    """
    Soft update w- weights from w.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    
    Params:
    - local_model (PyTorch model): local dqn_model
    - target_model (PyTorch model): target dqn_model
    """
    
    for target_w, local_w in zip(target_model.parameters(), local_model.parameters()):
      target_w.data.copy_((self.context['tau'] * local_w.data) + 
                          ((1.0 - self.context['tau']) * target_w.data)
                         )
      
    return
  
  def act(self, state, eps = 0.):
    """
    Returns an action for state as per the current policy
    
    Prams:
    - state (srray_like): current state
    - eps (float): epsilon, for epsilon-greedy action selection
    """
    
    state = torch.from_numpy(state).float().unsqueeze(0).to(self.context['device'])
    self.dqn_local.eval()
    with torch.no_grad():
      action_values = self.dqn_local(state)
      
    self.dqn_local.train()
    
    # epsilon-greedy action selection
    if random.random() > eps:
      a = np.argmax(action_values.cpu().data.numpy())
    else:
      a = random.choice(np.arange(self.context["action_size"]))
      
    return a
  
  
class PriorityReplayBuffer():
  """
  Fixed-size priority buffer to store experience tuples
  
  Params:
  - action_size (int): number of a in A
  - buffer_size (int): size of the buffer
  - batch_size (int): training batch size
  - seed (int): random seed
  - probability alpha (float): a number in the range [0, 1] 
  """
  
  def __init__(self, buffer_size, batch_size, seed, prob_alpha = 0.2, device = "cpu"):
    
    # self.action_size = action_size
    self.buffer_size = buffer_size
    self.memory = deque(maxlen = buffer_size)
    self.priority = np.array([])
    self.index = np.array(range(buffer_size))
    self.prob_alpha = prob_alpha
    self.batch_size = batch_size
    self.experience = namedtuple("Experience", field_names = ['state', 'action', 
                                                              'reward', 'next_state',
                                                              'done'])
    self.seed = seed
    self.device = device
    self.rng = np.random.default_rng(seed)
    
    return
  
  def insert(self, state, action, reward, next_state, done, p):
    """
    insert experience into memory
    
    Params:
      state (object): the state value
      action (object): the action value
      reward (object) the reward value
      next_state (object): the next state value
      done (boolean): done value
      p (float): priority value
    """

    e = self.experience(state, action, reward, next_state, done)
    self.memory.append(e)
    if self.priority.size < self.buffer_size:
      self.priority = np.append(self.priority, p)
    else:
      self.priority = np.append(self.priority[1:], p)
    
    return
  
  def update_priority(self, priorities, index):
    """
    updates priorities
    
    Params:
    - priorities (iterable): list of new priority values
    - index (iterable): the buffer location of the experiences to update
    """
    
    # for i in index:
    #   self.priority[i] = priorities[i]
    self.priority[index] = priorities
    
    return
  
  def sample(self, beta):
    """
    Random sample a batch of experiences from memory
    """
    
    p_i = self.priority ** self.prob_alpha
    p = (p_i / p_i.sum())
    sample_index = self.rng.choice(self.index[:self.priority.size], 
                             size = self.batch_size, p = p, shuffle = False)
    importance_w = (len(self.memory) * p[sample_index]) ** (-beta)
    
    
    states = torch.from_numpy(np.vstack([self.memory[i].state for i in sample_index])
                             ).float().to(self.device)
    
    actions = torch.from_numpy(np.vstack([self.memory[i].action for i in sample_index])
                              ).long().to(self.device)
    
    rewards = torch.from_numpy(np.vstack([self.memory[i].reward for i in sample_index])
                              ).float().to(self.device)
    
    next_states = torch.from_numpy(np.vstack([self.memory[i].next_state for i in sample_index])
                                  ).float().to(self.device)
    
    dones = torch.from_numpy(np.vstack([self.memory[i].done for i in sample_index]).astype(np.uint8)
                             ).float().to(self.device)
    
    return (states, actions, rewards, next_states, dones, sample_index, importance_w)
  
  def __len__(self):
    """
    return the size of the internal memory
    """
    return len(self.memory)
    

             