[//]: # (Image Ref)

[image1]: https://video.udacity-data.com/topher/2018/June/5b1ab4b0_banana/banana.gif "Banana Env"

# Udacity - Deep Reinforcement Learning
## Project 1 - Navigation: The Banana project

![Banana Env][image1]

This repository contains my submission to the course's Project 1

### Table of Contents

- [The Project](#The_Project)
- [The Environment](#The_Environment)
- [The Setup](#The_Setup)
- [The Agents](#The_Agents)

### The Project

Design, Implement, and train a Deep Q-Netwok agent to navigate (and collect bananas!) in a large, square world.

### The Environment
The state space has 37 dimensions which contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

> The **goal** of the agent is to collect as many yellow bananas as possible, and avoid collecting blue bananas.

The environment returns a reward of +1 for collecting a yellow banana, and a reward of -1 for collecting a blue banana.  The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### The Setup
The following instructions describe how to setup the computing environment to run this project implementation

#### Install and Configure the computing environment requirements

> Suggestion:  use [Conda](https://docs.conda.io/projects/conda/en/stable/), or your preferred virtual computing environment management system, to create a virtual computing environment for this project

Setup a computentional environment with the **3.6** version of _python_.  Then pip install the packages listed in this project's [requirements.txt](requirements.txt) file

#### Configure the Training Environment

1. Clone this project to your computing environment
> This repo doesn't include the banana.app, so please be sure and complete steps 2 and 3

2. Download the Banana Environment that matches your operating system from one of the links below:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
- (_For AWS_) If you'd like to train the agent on AWS please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.    

3. Place the file in the `p1_navigation/` directory, and unzip (or decompress) the file.

### The Agents
This project trains three different Deep Q-Network (DQN) based agents, each of which implements a different learning strategy, and compares their training performance.  The learning strategies implemented in this project are:
1. Fixed Q-Targets
2. Experience Replay
3. Double Q-Learning
4. Prioritized Experience Replay

Code for each agent is in the `pi_navigation/agent` directory, and the file nomenclature used to identify each implementation uses the item number of the learning strategy list above

- **dqn_agent_1_2**: Implements Fixed Q-Targets and Experience Replay
- **dqn_agent_1_2_3**: Implements Fixed Q-Targets, Experience Replay, and Double Q-Learning
- **dqn_agent_1_3_4**: Implements Fixed Q-Targets, Prioritized Experience Replay, and Double Q-learning

Learn more about the training process and results by reading the [README](p1_navigation/README.md) in the `p1_navigation` directory

