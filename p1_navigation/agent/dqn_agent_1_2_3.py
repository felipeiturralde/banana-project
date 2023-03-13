

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
  
  def __init__(self, state_size, action_size, seed, **kwargs):
    
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
    
    # DQ-network
    self.dqn_local = dqn_model.DQNetwork(state_size, action_size, seed).to(self.context['device'])
    self.dqn_local.apply(self.dqn_local.init_weights)
    
    self.dqn_target = dqn_model.DQNetwork(state_size, action_size, seed).to(self.context['device'])
    self.dqn_target.apply(self.dqn_target.init_weights)
    
    self.optimizer = optim.Adam(self.dqn_local.parameters(), lr=self.context['lr'])
    
    # Experience Replay Memory
    self.replay_buffer = ReplayBuffer(buffer_size = self.context['buffer_size'], 
                                      batch_size = self.context['batch_size'], 
                                      seed = seed, 
                                      device = self.context['device']
                                     )
    
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
  
  def step(self, state, action, reward, next_state, done):
    """
    Implements a DQN actions taken at each step
    """
    
    # store experience in replay buffer
    self.replay_buffer.insert(state, action, reward, next_state, done)
    
    # Learn
    self.t_step = (self.t_step + 1) % self.context['update_rate']
    if self.t_step == 0:
      if len(self.replay_buffer) > self.context['batch_size']:
        experiences = self.replay_buffer.sample()
        self.learn(experiences)
      
    return
  
  def learn(self, experiences):
    """
    Learn.
    i.e. update value parameters using experiences
    
    Params:
    - experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
    """
    
    states, actions, rewards, next_states, dones = experiences
    
    # get expected (current) Q values for these states from w model (dqn_local)
    Q_current = self.dqn_local(states).gather(1, actions)

    # Using Double Q-learning: 
    ## get best current w for q(S', a) = max(S', a, w)
    Q_current_next_best_actions = self.dqn_local(next_states).detach().max(1)[1].unsqueeze(1)
    ## Evaluate
    ## get best target w for q'(S', best_actions, w-)
    Q_target_next_best_actions = self.dqn_target(next_states).gather(1, 
                                                                     Q_current_next_best_actions
                                                                    )

    # compute the Q_targets:  If the episode is done we only keep the reward
    ## with double Q-learning w- dampers the impact of optimistic w
    Q_targets = rewards + (self.context['gamma'] * Q_target_next_best_actions * (1 - dones))
    
    # compute the error (loss)
    loss = F.mse_loss(Q_current, Q_targets)
    # minimize the loss
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    
    # update target network
    self.soft_update(local_model = self.dqn_local, target_model = self.dqn_target)
    
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
  
  
class ReplayBuffer():
  """
  Fixed-size buffer to store experience tuples
  
  Params:
  - action_size (int): number of a in A
  - buffer_size (int): size of the buffer
  - batch_size (int): training batch size
  - seed (int): random seed
  """
  
  def __init__(self, buffer_size, batch_size, seed, device = "cpu"):
    
    # self.action_size = action_size
    self.memory = deque(maxlen = buffer_size)
    self.batch_size = batch_size
    self.experience = namedtuple("Experience", field_names = ['state', 'action', 
                                                              'reward', 'next_state', 
                                                              'done'])
    self.seed = seed
    self.device = device
    
    return
  
  def insert(self, state, action, reward, next_state, done):
    """
    insert experience into memory
    """

    e = self.experience(state, action, reward, next_state, done)
    self.memory.append(e)
    
    return
  
  def sample(self):
    """
    Random sample a batch of experiences from memory
    """
    
    experiences = random.sample(self.memory, k = self.batch_size)
    
    states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])
                             ).float().to(self.device)
    actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])
                              ).long().to(self.device)
    rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])
                              ).float().to(self.device)
    next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])
                                  ).float().to(self.device)
    dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
                             ).float().to(self.device)
    
    return (states, actions, rewards, next_states, dones)
  
  def __len__(self):
    """
    return the size of the internal memory
    """
    return len(self.memory)
    

             