

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class DQNetwork(nn.Module):
  """
  Agent policy model
  Params:
  - state_size (int): number of s in S
  - action_size (int): number of a in A
  - seed (int): random seed
  """
  
  def __init__(self, state_size, action_size, seed):
    
    super(DQNetwork, self).__init__()
    self.seed = torch.manual_seed(seed)
    
    # Deep Network Architecture
    fc1_units = 128
    fc2_units = 128
    
    self.fc1 = nn.Linear(state_size, fc1_units)
    self.fc2 = nn.Linear(fc1_units, fc2_units)
    self.fc3 = nn.Linear(fc2_units, action_size)
    
    return
  
  def forward(self, state):
    """
    run the network
    """
    x = F.relu(self.fc1(state))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    
    return x
  
  def init_weights(self, m):
    """
    Initializes the layer weights using the general rule
    """
    classname = m.__class__.__name__
    if classname.find('linear') != -1:
      # get the number of inputs
      n = m.in_featuers
      y = 1.0 / np.sqrt(n)
      m.weight.data.uniform_(-y, y)
      m.bias.data.fill_(0)
      
    return
  