# Report: The Banana Project
---
For this project I implemented four learning improvements to the standard Deep Q-Learning method.  I used this improvements in different combinations to implement three different agens, and trained five instances of these agents to solve the P1 challenge.

## Table of contents
- [Implemented improvements](Implemented improvements)
- [DNN model architecture](DNN_model_architecture)
- [The Agents](The_Agents)
- [Results](Results)

## Implemented improvements
### 1. Fixed Q-Targets
At each stage of optimization the parameters from the previous **_n_** iterations **($\omega^{-}$)** are held fixed.  This strategy creates a more stable environment for agent learning, as $\omega^{-}$ remains fixed through **_n_** number of iterations, allowing the agent to experience an unchanging world.  After the _n_ iterations, we update the previous weights with the recently learned weights __($\omega^{-}$ := $\omega$)__, and resume learning for another _n_ iterations

$$\Delta\omega = \alpha \cdot (r + \gamma \max_{a'}{\hat{q}(s', a'; \omega^{-})} - \hat{q}(s, a; \omega)) \nabla_{\omega}\hat{q}(s, a, \omega)$$

### 2. Experience Replay
To prevent correlation between experiences to influence the agent's learning, _Experience Replay_ separates the process of _Interacting_ with the environment from the process of _Learning_.  During _Interaction_ experiences are stored in a _replay buffer_.  During _Learning_ a sample of experiences is randomly selected from the _replay_buffer_ for the agent to learn from.  The random sample of experiences from the _replay buffer_ helps the agent avoid being swayed by the correlation that exists between sequential experiences

### 3. Double Q-Learning
To damper an overoptimism of a pure _argmax_ strategy in learning Q-values, _Double Q-Learning_ splits the selection of the next best action and its evaluation during learning
- Next best action is selected as the _argmax_ of the most recently learned set of weights __($\omega$)__
- The value of the selected action is evaluated against the a fixed-value set of weights __($\omega^{-}$)__

After _n_ iterations, we update the fixed weights with the recently learned weights __($\omega^{-}$ := $\omega$)__, and resume learning for another _n_ iterations

$$\Delta\omega = \alpha \cdot (r + \gamma \hat{q}(s', \hat{q}(s', \max_{a'}{\hat{q}(s', a'; \omega)}); \omega^{-}) - \hat{q}(s, a; \omega)) \nabla_{\omega}\hat{q}(s, a, \omega)$$

### 4. Prioritized Experience Replay
In a standard Deep Q-Learning implementation, or one with the _Experience Replay_ improvement, infrequent experiences get a small opportunity to teach the agent.  A _Prioritized Experience Replay_ allows infrequent experiences to influence learning by bumping up their probability to be selected in a learning sample of experiences.  A _Prioritized Experience Replay_ is implemented by making three changes
- A probability is assigned to each experience during the interaction phase and updated during the Learning phase.  The implementation in this project uses the __TD error delta__ ($\delta_{t}$) as the priority for the experience, the bigger the error, the more the experience can teach the agent
$$\delta_{t} = r + \gamma \max_{a'}{\hat{q}(s', a'; \omega)} - \hat{q}(s, a; \omega) + e$$
where $e$ is a small constant added to prevent experiences with $\delta_{t} \approx 0$ from being starved of selection

- _Experience Replay_ sampling probability is updated from _uniform_ to _priority_ where the probability is computed from priorities when creating batches as
$$P_i = \frac{P_{i}^{a}}{\sum_{k}{P_{k}^a}}$$
where the hyperparameter $a$ allows the computation to be set between fully _uniform probability_ ($a := 0$), and fully _priority_ ($a := 1$)

- Add an _Importance-sampling weight_ term to the update rule in order to correct for any bias towards the priority values by amending the expectation over all experiences term
$$\Delta\omega = \alpha \cdot (\frac{1}{N} \cdot \frac{1}{Pi})^{b}\delta_t\nabla_\omega \hat{q}(s, a; \omega)$$
where $N$ is the __size__ of the _replay buffer_, and $b$ is a hyperparameter to control how much the weight affects learning

## DNN model architecture
A fully connected 3 layer DNN with 128 unitsis architecture is [implemented for all agents](model/dqn_model.py).  Internal layers are _ReLu_ activated and weights ($\omega$) are initialized using the _genral rule_

## The Agents
I implemented three different agents using a combination of the improvements described above:
- **dqn_agent_1_2**: Implements Fixed Q-Targets and Experience Replay
- **dqn_agent_1_2_3**: Implements Fixed Q-Targets, Experience Replay, and Double Q-Learning
- **dqn_agent_1_3_4**: Implements Fixed Q-Targets, Prioritized Experience Replay, and Double Q-learning

## Training
I trained five different instances with the following settings: 

| Agent | sessions | episodes | interactions per episode | $\epsilon$ start | $\epsilon$ decay | replay $a$ | replay $b$ | $e$ |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
|dqn_agent_1_2|5|2500|1000|1.0|0.995|n/a|n/a|1e-5|
|dqn_agent_1_2_3|5|2500|1000|1.0|0.995|n/a|n/a|1e-5|
|dqn_agent_1_3_4_005|5|2500|1000|1.0|0.995|0.05|0.4|1e-5|
|dqn_agent_1_3_4_01|5|2500|1000|1.0|0.995|0.1|0.4|1e-5|
|dqn_agent_1_3_4_02|5|2500|1000|1.0|0.995|0.2|0.4|1e-5|

## Results
All agents were trained 5 different times (_sessions = 5_) and the _learning speed_ was computed as the average number of learning episodes over all sessions:
$$learning\_speed = \frac{\sum_{s = 1}^{5} total\_episodes_s}{sessions}$$
All agents achieved a _learning_speed_ < 700, and while _dqn_agent_1_2_3_ was the fastest learner, it seems like _dqn_agent_1_3_4_01_ has a bigger learning potential; as described by the learning rate graph.  
The three fastest agents are:
1. dqn_agent_1_2_3
2. dqn_agent_1_3_4_005
3. dqn_agent_1_3_4_01

The learned weights of each agent are stored in the [learning_data](learning_data/) folder

Please see the [Analysis](Analysis.ipynb) notebook for a breakdown of the learning data.

### Future work
- Validate the performance hypothesis - agent **_dqn_agent_1_3_4_01_** is probably better at navigating the env - by conducting a competition against the other two agents and evaluating the results

- Submit agent **_dqn_agent_1_3_4_** to an expanded training with more variations of $a$ and $b$, to see if it's performance on this env can be fine-tuned to exceed that of **_dqn_agent_1_2_3_**

- Validate that the selected DNN architecture facilitates learning this env by training the agents on other architectures.  Starting on one with a smaller number of units for example
