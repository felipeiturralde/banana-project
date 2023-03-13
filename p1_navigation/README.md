# Navigation: The Banana Project
---
This project trains three different Deep Q-Network (DQN) based agents, each of which implements a different learning strategy, and compares their training performance.  The learning strategies implemented in this project are:
1. Fixed Q-Targets
2. Experience Replay
3. Double Q-Learning
4. Prioritized Experience Replay

Code for each agent is in the `pi_navigation/agent` directory, and the file nomenclature used to identify each implementation uses the item number of the learning strategy list above

- **dqn_agent_1_2**: Implements Fixed Q-Targets and Experience Replay
- **dqn_agent_1_2_3**: Implements Fixed Q-Targets, Experience Replay, and Double Q-Learning
- **dqn_agent_1_3_4**: Implements Fixed Q-Targets, Prioritized Experience Replay, and Double Q-learning

## Table of Contents
- [Training the agents](#training-the-agents)
  - [The Learning Process](#the_learning_process)
  - [The Training Process](#the_training_process)
- [Analysis](#analysis)
- [Report](#report)

## Training the agents
### The Learning Process
Each agent learns by interacting with the environment for as many episodes as are necessary for the agent to achieve a +13 mean score over 100 consecutive episodes.

The [Train_Agent](agent/training.py) class orchestrates the agents learning process by serving as the agent-environment interface.  The Train method sets up a training session for an agent.  It controls for how many episodes the agent is allowed to learn and how many discrete time steps make up an episode.  It also has defaults for target_score, and $\epsilon$-greedy and priority replay $a$ learning hyperparameters

```
  def Train(self, n_episodes = 2500, max_t = 1000, target_score = np.inf, 
            eps_start = 1.0, eps_end = 0.01, eps_decay = 0.995, 
            priority_replay = False, exp_replay_alpha = 0.2):
    """
    Trains the agent, and records training performance
    
    The agent is trained for n_episodes or untill it reaches target_score.
    Each episode lasts max_t interactions.
    
    The function keeps tract of 
    
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
    """

```

### The Training Process
The `run` function in the [Navigation](Navigation.ipynb) notebook implements a training session for a set of agents.  
> The **Training one agent** section in the Navigation notebook shows how to train one agent for one session

To train the agents, after correctly configuring the computing environment (see [The Setup](../#the-setup) section in the project README file for setup instructions), execute the cells under the `Initialize notebook setings` and `Training Process` sections.  You can then select to `Train one agent`, or `Train all agents`.

## Analysis
Please see the [Analysis](Analysis.ipynb) notebook for a breakdown of the learning data.

## Report
Please see the [Report](Report.md) file for a description of the implemented Deep Q-Learning improvements