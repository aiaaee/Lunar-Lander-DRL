# Lunar Lander with DQN and Double DQN

## Project Overview 

This project implements and evaluates **Deep Q-Network (DQN)** and **Double Deep Q-Network (Double DQN)** to solve the **Lunar Lander** environment. The objective of the task is to train an autonomous agent to perform a safe and stable landing by maximizing the cumulative reward through interaction with the environment. The project demonstrates how value-based reinforcement learning algorithms can learn an effective landing policy from trial-and-error experience.

The implementation is developed entirely in **Python** using **PyTorch** for the neural network components, while **NumPy** and native Python are used to build the reinforcement learning pipeline, including the agent, replay buffer, and training process. The DQN implementation incorporates several key techniques such as **Experience Replay**, **Target Networks**, and an **ε-greedy exploration strategy** to improve training stability and learning efficiency.

The primary goal of this project is to compare the performance of DQN and Double DQN under the same environment and training conditions. In addition to the performance comparison, the project aims to provide a clear understanding of agent behavior during learning and to illustrate the transition from classical tabular reinforcement learning concepts to deep reinforcement learning methods through a clean, modular implementation.

<img width="600" height="400" alt="lunar_lander" src="https://github.com/user-attachments/assets/3eecd699-c893-41fe-b7c9-a03e578af796" />

## Demo 

The following demonstrations show the trained agents interacting with the Lunar Lander environment after completing the training process.

https://github.com/user-attachments/assets/85559064-dfa9-4437-b5fd-815a02868c7b

https://github.com/user-attachments/assets/b3394d80-0ccb-412b-baf3-8340d537ffc2



## Features 

- Implementation of Deep Q-Network (DQN) from scratch using PyTorch  
- Implementation of Double DQN (DDQN) to reduce overestimation bias  
- Experience Replay for stable training  
- Target Network for improved convergence  
- Epsilon-Greedy policy for exploration vs exploitation  
- Modular and clean project structure  
- Training and evaluation pipelines separated  
- Performance comparison between DQN and Double DQN  


## Project Structure

```text
Lunar-Lander-RL/
├── configs/
│   ├── dqn.yaml
│   └── double_dqn.yaml
│
├── src/
│   ├── agents/
│   │   ├── dqn_agent.py
│   │   └── double_dqn_agent.py
│   │
│   ├── networks/
│   │   └── q_network.py
│   │
│   ├── memory/
│   │   └── replay_buffer.py
│   │
│   ├── training/
│   │   ├── train_dqn.py
│   │   └── train_double_dqn.py
│   │
│   ├── evaluation/
│   │   └── evaluate.py
│   │
│   └── utils/
│       ├── plotting.py
│       ├── logger.py
│       └── seed.py
│
├── models/
│   ├── dqn/
│   └── double_dqn/
│
├── results/
│   ├── plots/
│   ├── videos/
│   └── logs/
│
├── README.md
├── requirements.txt
└── LICENSE
```

The project follows a modular structure that separates the agent, neural network, replay buffer, training pipeline, evaluation, and utility functions, making the codebase easier to maintain and extend.


## Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Lunar-Lander-RL.git
cd Lunar-Lander-RL
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the agent

To train the DQN agent:

```bash
python src/training/train_dqn.py
```

To train the Double DQN agent:

```bash
python src/training/train_double_dqn.py
```

### 4. Evaluate a trained model

```bash
python src/evaluation/evaluate.py
```

## Result 

To evaluate the performance of the DQN and Double DQN agents on the LunarLander-v2 environment, both models were trained under identical experimental conditions for 2,000 episodes. For visualization purposes, the reward curves were plotted using 800 episodes, providing a clearer representation of the learning trends.

The experimental results indicate that Double DQN achieved superior performance compared to the standard DQN model. Specifically, the DQN agent obtained an average reward of 245.3 and a best reward of 302.1. In contrast, the Double DQN agent achieved an average reward of 281.7 and a best reward of 319.8.


<img width="583" height="455" alt="Double DQN - Lunar Lander (Main)" src="https://github.com/user-attachments/assets/a343c082-eaa7-465f-8bae-d73b21741c5c" />


The reward-versus-episode curves further support these findings, showing that the Double DQN agent demonstrated more stable learning behavior and reached higher reward values during training. The improved performance can be attributed to the Double DQN mechanism, which reduces the overestimation bias commonly observed in standard DQN by separating action selection from action evaluation.

Overall, the results demonstrate that Double DQN is more effective than DQN for the Lunar Lander task, achieving higher rewards and exhibiting more stable learning dynamics under the same training conditions.

## Future Work

This project provides a baseline implementation of DQN and Double DQN for the Lunar Lander environment. Future improvements may include:

* Implementing **Dueling DQN** to improve value estimation.
* Integrating **Prioritized Experience Replay (PER)** for more efficient sampling.
* Extending the project with advanced reinforcement learning algorithms such as **Rainbow DQN** and **PPO**.
* Performing comprehensive hyperparameter tuning and performance evaluation.
* Comparing the implemented algorithms using quantitative metrics and training curves.
* Improving the visualization of the agent's behavior with additional demonstrations and analysis.


