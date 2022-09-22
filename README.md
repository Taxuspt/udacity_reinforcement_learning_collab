# Udacity RL course: Project 3 : Continuous control
![robot](images/tennis_agent.gif)

## Project details
Solution for the third project of Udacity's course on Reinforcement Learning.
The goal of the agent is to play a Tennis game against other agent.

The simulation environment was created in Unity and is available [here](envs/Tennis.app) (macOS version).
```
INFO:unityagents:
'Academy' started successfully!
Unity Academy name: Academy
        Number of Brains: 1
        Number of External Brains : 1
        Lesson number : 0
        Reset Parameters :
		
Unity brain name: TennisBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 8
        Number of stacked Vector Observation: 3
        Vector Action space type: continuous
        Vector Action space size (per agent): 2
        Vector Action descriptions: , 
```
After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores. This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Getting started
This project was developed on an Apple Macbook with an M1 chip, therefore the installation of Python packages is often cumbersome. 
The setup script assumes that you have a compatible [conda](https://conda-forge.org/blog/posts/2020-10-29-macos-arm64/) installation and python 3.8 installed on your system.

To create a new conda environment and install the necessary packages, run:
```
source setup.sh
```

You should be able to run the imports on [Tennis](Tennis.ipynb) without any error.

## Contents
`Tennis.ipynb` - Notebook implementing the multi-agent Deep Deterministic Policy Gradient (MA-DDPG) Actor-Critic algorithm. Allows you to train and test your agents.

`agent.py` - The agent that interacts and learns from the environment.

`model` - The neural networks that model the Critic and Actor.


## Training and running the agent
Run [Tennis](Tennis.ipynb) to train and test an agent.

You should be able to get an average score >0.5 after ~5000 episodes, although this varies quire a lot.
You can also load a previously trained agent directly, without having to run the training job.
