# WTF 深度强化学习教程 1. Deep Q-Network

WTF 深度强化学习教程，帮助新人快速入门 Deep RL，算法使用pytorch 2.0版本实现。

**推特**：[@WTFAcademy_](https://twitter.com/WTFAcademy_) ｜ [@0xAA_Science](https://twitter.com/0xAA_Science)

**WTF Academy 社群：** [官网 wtf.academy](https://wtf.academy/) | [WTF Solidity 教程](https://github.com/AmazingAng/WTFSolidity) | [discord](https://discord.gg/5akcruXrsk/) | [微信群申请](https://docs.google.com/forms/d/e/1FAIpQLSe4KGT8Sh6sJ7hedQRuIYirOoZK_85miz3dw7vA1-YjodgJ-A/viewform?usp=sf_link)

所有代码和教程开源在 github: [github.com/AmazingAng/WTF-DeepRL](https://github.com/WTFAcademy/WTF-DeepRL)

---

这一讲，我们将尝试利用pytorch实现深度强化学习的开山之作 Deep Q-Network，DQN，推荐你先阅读 [DQN 论文](https://arxiv.org/abs/1312.5602)。

## 0. 先修课程

在开始之前，你需要先完成先修课程：

1. 经典强化学习理论：推荐 Sutton 和 Barto 写的[强化学习圣经](http://incompleteideas.net/book/RLbook2020.pdf)。

    ![](./img/1-1.png)

2. 机器学习：你可以在网上找到很多机器学习的公开课，比如coursera上Andrew Ng的课程。

3. python编程：网上你可以找到很多的python入门公开课，这里推荐哈佛大学的CS50 python版。

## 1. 深度强化学习中的元素

强化学习研究的是智能体（Agent）和环境（Environment）交互中如何学习最优策略，以获得最大收益（Cumulative rewards）。Agent需要能够观察环境(observe)的到所处的状态，评判（value）状态下每个动作的价值，选出最优的动作（act）来和环境交互，同时通过从经验中学习不断改善自己的策略（learn from experience）。因此，observe，value，act和learn是强化学习Agent必不可少的元素。

![](./img/1-2.png)

如果我们给Agent写一个类，大体会长这样的：

```python
class Agent: 

    def __init__(self):
        ...

    def observe(self, observation):
        ...
        return state

    def value(self, state,):
        ...
        return value_of_actions
    
    def act(value_of_actions):
        ...
        return selected_action
    
    def learn_from_experience(self, batch_size):
        ...
```

这个教程中，我们会使用经典的Atari游戏来训练强化学习算法，下面我们探讨一下这几个函数在Atari环境中起到什么作用：

- `observe`: 在Atari中，环境每一步给出的observation（84x84x1的array）可以直接作为state。那么observe()函数只需要把numpy array转换为torch tensor，方便模型后续使用就好了。在更复杂的partial observable环境，我们需要利用observation来推断所处的state，这时observe()函数会由更多功能。
- `value`: 在DQN中，`value`函数主要是给出当前state下每个action的Q value，帮助智能体选择最优策略。
- `act`: 在DQN中，根据`value`函数给出的Q值，采用epsilon greedy policy选出action。
- `learn_from_experience`: 根据收集的经验计算TD Loss（temporal-difference loss），再通过梯度下降算法更新深度神经网络的参数，改善策略。其中TD Loss由Bellman Equation给出：

    $$Loss_{TD}=R_t+\gamma Q(s_{t+1},a_{t+1})−Q(s_t,a_t)$$

下面，我们开始完成DQN算法。

## 2. 引入包
你需要安装相应的包，然后在 jupyter notebook 中导入他们，如果你使用的是Google Colab Research，则需要安装`gym[atari]`和`autorom[accept-rom-license]`。


```python
# 在 Google Colab Rsearch 中需要安装的库
# !pip install gym[atari]
# !pip install autorom[accept-rom-license]

import gym, random, pickle, os.path, math, glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import pdb

from gym.wrappers import AtariPreprocessing, LazyFrames, FrameStack

from IPython.display import clear_output
from torch.utils.tensorboard import SummaryWriter

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
```

## 3. Atari游戏中的Pong

Pong是Atari中一个仿真打乒乓球的游戏：玩家和电脑每人拿一个板子，接对方弹来的球，没接住的话，对方得一分，先得到21分的获胜。

我们使用 DQN 论文中的设定，在丢失声明时会结束游戏，并且4帧画面会合并为1个输入，加快学习。


```python
# Create and wrap the environment
env = gym.make('PongNoFrameskip-v4')
env = AtariPreprocessing(env,
                         scale_obs=False,
                         terminal_on_life_loss=True,
                         )
env = FrameStack(env, num_stack=4)
n_actions = env.action_space.n
state_dim = env.observation_space.shape

# env.render()
test = env.reset()
for i in range(100):
    test = env.step(env.action_space.sample())[0]

plt.imshow(test.__array__()[0,...])

# env.close()
```

    
![png](DQN_files/DQN_4_2.png)
    


## 4. Deep-Q Network

对于复杂的问题，state维度非常大，我们很难基于tabular method来判断每一个(state, action)的价值。这种情况下，我们利用function approximation方法，构建一个深度神经网络(Deep-Q Network, DQN)，来估计(state, action)的价值。value()中Deep-Q Network模块就是一个神经网络，输入是atari game中的一帧图像，输出是每个action的价值。


```python
class DQN(nn.Module):
    def __init__(self, in_channels=4, num_actions=5):
        """
        Initialize a deep Q-learning network as described in
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.reshape(x.size(0), -1)))
        return self.fc5(x)

```

## 5. Memory
因为深度神经网络收敛很慢，需要非常多的样本，如果只根据环境交互来训练网络，将非常的没效率。因此DQN引入了一个memory buffer来进行memory replay，就是把之前和环境交互的经验存下来，在训练时重复利用。memory buffer主要实现两个函数：`push`函数将经验存入，`sample`函数将经验取出用于训练。


```python
class Memory_Buffer(object):
    def __init__(self, memory_size=100000):
        self.buffer = []
        self.memory_size = memory_size
        self.next_idx = 0

    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if len(self.buffer) <= self.memory_size: # buffer not full
            self.buffer.append(data)
        else: # buffer is full
            self.buffer[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % self.memory_size

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in range(batch_size):

            idx = random.randint(0, self.size() - 1)
            data = self.buffer[idx]
            state, action, reward, next_state, done= data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones

    def size(self):
        return len(self.buffer)

```

## 6. Agent

下面，我们要写最复杂的部分，实现基于DQN的智能体。我们分别实现了下列函数：

- `__init__`: 初始化DQN智能体的参数和网络。
- `observe`: 将Atari环境每一步返回的observation（numpy矩阵）转为状态（pytorch tensor）。
- `value`: 返回状态的Q值。
- `act`: 给定状态，根据epsilon greedy算法给出当前动作。
- `sample_from_buffer`: 学习相关，从memory buffer抽样经验。
- `compute_td_loss`: 学习相关，利用从memory buffer抽样的经验计算TD Loss。
- `learn_from_experience`: 学习相关，利用TD Loss进行梯度下降，优化网络。


```python
class DQNAgent:
    def __init__(self, in_channels = 1, action_space = [], USE_CUDA = False, memory_size = 10000, epsilon  = 1, lr = 1e-4):
        self.epsilon = epsilon
        self.action_space = action_space
        self.memory_buffer = Memory_Buffer(memory_size)
        self.DQN = DQN(in_channels = in_channels, num_actions = action_space.n)
        self.DQN_target = DQN(in_channels = in_channels, num_actions = action_space.n)
        self.DQN_target.load_state_dict(self.DQN.state_dict())

        self.USE_CUDA = USE_CUDA
        if USE_CUDA:
            self.DQN = self.DQN.cuda()
            self.DQN_target = self.DQN_target.cuda()
        self.optimizer = optim.RMSprop(self.DQN.parameters(),lr=lr, eps=0.001, alpha=0.95)

    def observe(self, lazyframe):
        # from Lazy frame to tensor
        state =  torch.from_numpy(lazyframe.__array__()[None]/255).float()
        if self.USE_CUDA:
            state = state.cuda()
        return state

    def value(self, state):
        q_values = self.DQN(state)
        return q_values

    def act(self, state, epsilon = None):
        """
        sample actions with epsilon-greedy policy
        recap: with p = epsilon pick random action, else pick action with highest Q(s,a)
        """
        if epsilon is None: epsilon = self.epsilon

        q_values = self.value(state).cpu().detach().numpy()
        if random.random()<epsilon:
            aciton = random.randrange(self.action_space.n)
        else:
            aciton = q_values.argmax(1)[0]
        return aciton

    def compute_td_loss(self, states, actions, rewards, next_states, is_done, gamma=0.99):
        """ Compute td loss using torch operations only. Use the formula above. """
        actions = torch.tensor(actions).long()    # shape: [batch_size]
        rewards = torch.tensor(rewards, dtype =torch.float)  # shape: [batch_size]
        is_done = torch.tensor(is_done).bool()  # shape: [batch_size]

        if self.USE_CUDA:
            actions = actions.cuda()
            rewards = rewards.cuda()
            is_done = is_done.cuda()

        # get q-values for all actions in current states
        predicted_qvalues = self.DQN(states)

        # select q-values for chosen actions
        predicted_qvalues_for_actions = predicted_qvalues[
          range(states.shape[0]), actions
        ]

        # compute q-values for all actions in next states
        predicted_next_qvalues = self.DQN_target(next_states) # YOUR CODE

        # compute V*(next_states) using predicted next q-values
        next_state_values =  predicted_next_qvalues.max(-1)[0] # YOUR CODE

        # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
        target_qvalues_for_actions = rewards + gamma *next_state_values # YOUR CODE

        # at the last state we shall use simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
        target_qvalues_for_actions = torch.where(
            is_done, rewards, target_qvalues_for_actions)

        # mean squared error loss to minimize
        #loss = torch.mean((predicted_qvalues_for_actions -
        #                   target_qvalues_for_actions.detach()) ** 2)
        loss = F.smooth_l1_loss(predicted_qvalues_for_actions, target_qvalues_for_actions.detach())

        return loss

    def sample_from_buffer(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in range(batch_size):
            idx = random.randint(0, self.memory_buffer.size() - 1)
            data = self.memory_buffer.buffer[idx]
            frame, action, reward, next_frame, done= data
            states.append(self.observe(frame))
            actions.append(action)
            rewards.append(reward)
            next_states.append(self.observe(next_frame))
            dones.append(done)
        return torch.cat(states), actions, rewards, torch.cat(next_states), dones

    def learn_from_experience(self, batch_size):
        if self.memory_buffer.size() > batch_size:
            states, actions, rewards, next_states, dones = self.sample_from_buffer(batch_size)
            td_loss = self.compute_td_loss(states, actions, rewards, next_states, dones)
            self.optimizer.zero_grad()
            td_loss.backward()
            for param in self.DQN.parameters():
                param.grad.data.clamp_(-1, 1)

            self.optimizer.step()
            return(td_loss.item())
        else:
            return(0)

```

## 7. Traning

接下来是最重要的训练部分，基本上就是定好初始参数，要训练的总帧数，然后让智能体与环境交互并学习。


```python
# if __name__ == '__main__':

# Training DQN in PongNoFrameskip-v4
env = gym.make('PongNoFrameskip-v4')
env = AtariPreprocessing(env,
                         scale_obs=False,
                         terminal_on_life_loss=True,
                         )
env = FrameStack(env, num_stack=4)

gamma = 0.99
epsilon_max = 1
epsilon_min = 0.05
eps_decay = 30000
frames = 2000000
USE_CUDA = False
learning_rate = 2e-4
max_buff = 100000
update_tar_interval = 1000
batch_size = 32
print_interval = 1000
log_interval = 1000
learning_start = 10000
win_reward = 18     # Pong-v4
win_break = True

action_space = env.action_space
action_dim = env.action_space.n
state_dim = env.observation_space.shape[1]
state_channel = env.observation_space.shape[0]
agent = DQNAgent(in_channels = state_channel, action_space= action_space, USE_CUDA = USE_CUDA, lr = learning_rate, memory_size = max_buff)

frame = env.reset()

episode_reward = 0
all_rewards = []
losses = []
episode_num = 0
is_win = False
# tensorboard
summary_writer = SummaryWriter(log_dir = "DQN_stackframe", comment= "good_makeatari")

# e-greedy decay
epsilon_by_frame = lambda frame_idx: epsilon_min + (epsilon_max - epsilon_min) * math.exp(
            -1. * frame_idx / eps_decay)
# plt.plot([epsilon_by_frame(i) for i in range(10000)])

for i in range(frames):
    epsilon = epsilon_by_frame(i)
    state_tensor = agent.observe(frame)
    action = agent.act(state_tensor, epsilon)

    next_frame, reward, done ,_ = env.step(action)

    episode_reward += reward
    agent.memory_buffer.push(frame, action, reward, next_frame, done)
    frame = next_frame

    loss = 0
    if agent.memory_buffer.size() >= learning_start:
        loss = agent.learn_from_experience(batch_size)
        losses.append(loss)

    if i % print_interval == 0:
        print("frames: %5d, reward: %5f, loss: %4f, epsilon: %5f, episode: %4d" % (i, np.mean(all_rewards[-10:]), loss, epsilon, episode_num))
        summary_writer.add_scalar("Temporal Difference Loss", loss, i)
        summary_writer.add_scalar("Mean Reward", np.mean(all_rewards[-10:]), i)
        summary_writer.add_scalar("Epsilon", epsilon, i)

    if i % update_tar_interval == 0:
        agent.DQN_target.load_state_dict(agent.DQN.state_dict())

    if done:

        frame = env.reset()

        all_rewards.append(episode_reward)
        episode_reward = 0
        episode_num += 1
        avg_reward = float(np.mean(all_rewards[-100:]))

summary_writer.close()
```

    /usr/local/lib/python3.10/dist-packages/numpy/core/fromnumeric.py:3474: RuntimeWarning: Mean of empty slice.
      return _methods._mean(a, axis=axis, dtype=dtype,
    /usr/local/lib/python3.10/dist-packages/numpy/core/_methods.py:189: RuntimeWarning: invalid value encountered in double_scalars
      ret = ret.dtype.type(ret / rcount)


    frames:     0, reward:   nan, loss: 0.000000, epsilon: 1.000000, episode:    0
    frames:  1000, reward: -20.000000, loss: 0.000000, epsilon: 0.968855, episode:    1
    frames:  2000, reward: -20.000000, loss: 0.000000, epsilon: 0.938732, episode:    1
    frames:  3000, reward: -20.000000, loss: 0.000000, epsilon: 0.909596, episode:    3
    frames:  4000, reward: -20.250000, loss: 0.000000, epsilon: 0.881415, episode:    4
    frames:  5000, reward: -19.800000, loss: 0.000000, epsilon: 0.854158, episode:    5
    frames:  6000, reward: -20.000000, loss: 0.000000, epsilon: 0.827794, episode:    6
    frames:  7000, reward: -20.142857, loss: 0.000000, epsilon: 0.802295, episode:    7
    frames:  8000, reward: -20.333333, loss: 0.000000, epsilon: 0.777632, episode:    9
    frames:  9000, reward: -20.400000, loss: 0.000000, epsilon: 0.753777, episode:   10
    frames: 10000, reward: -20.500000, loss: 0.015006, epsilon: 0.730705, episode:   11
    frames: 11000, reward: -20.600000, loss: 0.015339, epsilon: 0.708389, episode:   12
    frames: 12000, reward: -20.400000, loss: 0.030417, epsilon: 0.686804, episode:   13
    frames: 13000, reward: -20.400000, loss: 0.015504, epsilon: 0.665927, episode:   14
    frames: 14000, reward: -20.300000, loss: 0.000243, epsilon: 0.645735, episode:   15
    frames: 15000, reward: -20.000000, loss: 0.014063, epsilon: 0.626204, episode:   17
    frames: 16000, reward: -20.000000, loss: 0.015310, epsilon: 0.607314, episode:   18
    frames: 17000, reward: -19.900000, loss: 0.000627, epsilon: 0.589043, episode:   19
    frames: 18000, reward: -19.900000, loss: 0.015577, epsilon: 0.571371, episode:   20
    frames: 19000, reward: -19.700000, loss: 0.000137, epsilon: 0.554278, episode:   21
    frames: 20000, reward: -19.700000, loss: 0.000225, epsilon: 0.537746, episode:   22
    frames: 21000, reward: -19.800000, loss: 0.000304, epsilon: 0.521756, episode:   23
    frames: 22000, reward: -19.600000, loss: 0.000069, epsilon: 0.506290, episode:   24
    frames: 23000, reward: -19.900000, loss: 0.000526, epsilon: 0.491331, episode:   25
    frames: 24000, reward: -19.900000, loss: 0.011583, epsilon: 0.476863, episode:   26
    frames: 25000, reward: -20.000000, loss: 0.030550, epsilon: 0.462868, episode:   27
    frames: 26000, reward: -20.000000, loss: 0.000159, epsilon: 0.449333, episode:   28
    frames: 27000, reward: -20.100000, loss: 0.029523, epsilon: 0.436241, episode:   30
    frames: 28000, reward: -20.300000, loss: 0.000606, epsilon: 0.423579, episode:   31
    frames: 29000, reward: -20.200000, loss: 0.015362, epsilon: 0.411331, episode:   32
    frames: 30000, reward: -20.100000, loss: 0.000134, epsilon: 0.399485, episode:   33
    frames: 31000, reward: -20.300000, loss: 0.030215, epsilon: 0.388028, episode:   34
    frames: 32000, reward: -20.300000, loss: 0.000039, epsilon: 0.376946, episode:   35
    frames: 33000, reward: -20.500000, loss: 0.000711, epsilon: 0.366228, episode:   36
    frames: 34000, reward: -20.300000, loss: 0.044813, epsilon: 0.355860, episode:   37
    frames: 35000, reward: -20.200000, loss: 0.015289, epsilon: 0.345833, episode:   38
    frames: 36000, reward: -20.200000, loss: 0.046345, epsilon: 0.336135, episode:   40
    frames: 37000, reward: -20.100000, loss: 0.000864, epsilon: 0.326754, episode:   41
    frames: 38000, reward: -20.300000, loss: 0.015155, epsilon: 0.317681, episode:   42
    frames: 39000, reward: -20.400000, loss: 0.000658, epsilon: 0.308905, episode:   43
    frames: 40000, reward: -20.400000, loss: 0.000096, epsilon: 0.300417, episode:   44
    frames: 41000, reward: -20.500000, loss: 0.016093, epsilon: 0.292208, episode:   45
    frames: 42000, reward: -20.700000, loss: 0.015165, epsilon: 0.284267, episode:   47
    frames: 43000, reward: -20.800000, loss: 0.000402, epsilon: 0.276587, episode:   48
    frames: 44000, reward: -20.800000, loss: 0.000187, epsilon: 0.269159, episode:   49
    frames: 45000, reward: -20.800000, loss: 0.029316, epsilon: 0.261974, episode:   50
    frames: 46000, reward: -20.900000, loss: 0.014796, epsilon: 0.255024, episode:   51
    frames: 47000, reward: -20.900000, loss: 0.029547, epsilon: 0.248303, episode:   52
    frames: 48000, reward: -21.000000, loss: 0.060029, epsilon: 0.241802, episode:   53
    frames: 49000, reward: -20.800000, loss: 0.030703, epsilon: 0.235514, episode:   55
    frames: 50000, reward: -20.800000, loss: 0.000256, epsilon: 0.229432, episode:   56
    frames: 51000, reward: -20.800000, loss: 0.000333, epsilon: 0.223549, episode:   57
    frames: 52000, reward: -20.800000, loss: 0.001257, epsilon: 0.217860, episode:   58
    frames: 53000, reward: -20.800000, loss: 0.022521, epsilon: 0.212357, episode:   59
    frames: 54000, reward: -20.700000, loss: 0.007153, epsilon: 0.207034, episode:   60
    frames: 55000, reward: -20.500000, loss: 0.002334, epsilon: 0.201886, episode:   61
    frames: 56000, reward: -20.300000, loss: 0.002344, epsilon: 0.196906, episode:   62
    frames: 57000, reward: -20.400000, loss: 0.017378, epsilon: 0.192090, episode:   64
    frames: 58000, reward: -20.300000, loss: 0.007723, epsilon: 0.187432, episode:   65
    frames: 59000, reward: -20.300000, loss: 0.001621, epsilon: 0.182926, episode:   66
    frames: 60000, reward: -20.300000, loss: 0.004116, epsilon: 0.178569, episode:   67
    frames: 61000, reward: -20.300000, loss: 0.004020, epsilon: 0.174354, episode:   68
    frames: 62000, reward: -20.300000, loss: 0.005219, epsilon: 0.170277, episode:   69
    frames: 63000, reward: -20.600000, loss: 0.004307, epsilon: 0.166334, episode:   71
    frames: 64000, reward: -20.800000, loss: 0.001022, epsilon: 0.162520, episode:   72
    frames: 65000, reward: -20.800000, loss: 0.005565, epsilon: 0.158831, episode:   73
    frames: 66000, reward: -20.800000, loss: 0.000940, epsilon: 0.155263, episode:   74
    frames: 67000, reward: -20.900000, loss: 0.002426, epsilon: 0.151812, episode:   75
    frames: 68000, reward: -20.800000, loss: 0.002548, epsilon: 0.148474, episode:   77
    frames: 69000, reward: -20.800000, loss: 0.001696, epsilon: 0.145246, episode:   78
    frames: 70000, reward: -20.800000, loss: 0.012093, epsilon: 0.142123, episode:   79
    frames: 71000, reward: -20.600000, loss: 0.001570, epsilon: 0.139103, episode:   80
    frames: 72000, reward: -20.400000, loss: 0.001393, epsilon: 0.136182, episode:   81
    frames: 73000, reward: -20.400000, loss: 0.000825, epsilon: 0.133357, episode:   82
    frames: 74000, reward: -20.400000, loss: 0.000892, epsilon: 0.130624, episode:   83
    frames: 75000, reward: -20.400000, loss: 0.001577, epsilon: 0.127981, episode:   84
    frames: 76000, reward: -20.300000, loss: 0.002478, epsilon: 0.125424, episode:   85
    frames: 77000, reward: -20.400000, loss: 0.000952, epsilon: 0.122952, episode:   86
    frames: 78000, reward: -20.300000, loss: 0.002204, epsilon: 0.120560, episode:   87
    frames: 79000, reward: -20.300000, loss: 0.007062, epsilon: 0.118247, episode:   89
    frames: 80000, reward: -20.400000, loss: 0.001367, epsilon: 0.116009, episode:   90
    frames: 81000, reward: -20.500000, loss: 0.002477, epsilon: 0.113845, episode:   91
    frames: 82000, reward: -20.500000, loss: 0.000874, epsilon: 0.111752, episode:   92
    frames: 83000, reward: -20.400000, loss: 0.000912, epsilon: 0.109728, episode:   93
    frames: 84000, reward: -20.500000, loss: 0.001354, epsilon: 0.107770, episode:   94
    frames: 85000, reward: -20.600000, loss: 0.002631, epsilon: 0.105876, episode:   95
    frames: 86000, reward: -20.600000, loss: 0.003812, epsilon: 0.104044, episode:   96
    frames: 87000, reward: -20.700000, loss: 0.000584, epsilon: 0.102272, episode:   98
    frames: 88000, reward: -20.700000, loss: 0.002449, epsilon: 0.100558, episode:   98
    frames: 89000, reward: -20.700000, loss: 0.002212, epsilon: 0.098901, episode:   99
    frames: 90000, reward: -20.800000, loss: 0.002638, epsilon: 0.097298, episode:  100
    frames: 91000, reward: -20.900000, loss: 0.001160, epsilon: 0.095747, episode:  101
    frames: 92000, reward: -20.900000, loss: 0.003304, epsilon: 0.094247, episode:  102
    frames: 93000, reward: -21.000000, loss: 0.000919, epsilon: 0.092797, episode:  103
    frames: 94000, reward: -21.000000, loss: 0.000890, epsilon: 0.091394, episode:  104
    frames: 95000, reward: -21.000000, loss: 0.004785, epsilon: 0.090037, episode:  105
    frames: 96000, reward: -20.900000, loss: 0.004902, epsilon: 0.088724, episode:  106
    frames: 97000, reward: -20.900000, loss: 0.001434, epsilon: 0.087455, episode:  107
    frames: 98000, reward: -20.900000, loss: 0.003306, epsilon: 0.086227, episode:  108
    frames: 99000, reward: -20.700000, loss: 0.002157, epsilon: 0.085039, episode:  109
    frames: 100000, reward: -20.600000, loss: 0.001317, epsilon: 0.083890, episode:  110
    frames: 101000, reward: -20.400000, loss: 0.017545, epsilon: 0.082779, episode:  111
    frames: 102000, reward: -20.300000, loss: 0.003890, epsilon: 0.081705, episode:  112
    frames: 103000, reward: -20.200000, loss: 0.001678, epsilon: 0.080665, episode:  113
    frames: 104000, reward: -20.200000, loss: 0.000882, epsilon: 0.079660, episode:  114
    frames: 105000, reward: -20.200000, loss: 0.003738, epsilon: 0.078688, episode:  115
    frames: 106000, reward: -20.300000, loss: 0.000946, epsilon: 0.077747, episode:  116
    frames: 107000, reward: -20.300000, loss: 0.006670, epsilon: 0.076837, episode:  117
    frames: 108000, reward: -20.300000, loss: 0.000709, epsilon: 0.075958, episode:  118
    frames: 109000, reward: -20.500000, loss: 0.002426, epsilon: 0.075107, episode:  119
    frames: 110000, reward: -20.500000, loss: 0.003445, epsilon: 0.074283, episode:  120
    frames: 111000, reward: -20.700000, loss: 0.002423, epsilon: 0.073487, episode:  121
    frames: 112000, reward: -20.800000, loss: 0.001779, epsilon: 0.072717, episode:  122
    frames: 113000, reward: -20.800000, loss: 0.007298, epsilon: 0.071973, episode:  122
    frames: 114000, reward: -20.900000, loss: 0.004011, epsilon: 0.071252, episode:  123
    frames: 115000, reward: -20.800000, loss: 0.003043, epsilon: 0.070556, episode:  124


## 8. Results

下面我们使用matplotlib库画出游戏得分和loss曲线。

从左边的图我们可以看到，DQN智能体在玩了200把游戏后开始快速学习，大概在300把游戏之后学习成功（达到20+分）。

从右边的图我们可以看到，在表现达到最优之后，Loss还是在不断减小。


```python
def plot_training(frame_idx, rewards, losses):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()

plot_training(i, all_rewards, losses)
```


    
![png](DQN_files/DQN_14_0.png)
    

## 9. Summary

感想就是，总结强化学习需要的元素相对容易，真正实现的时候很麻烦，尤其是当模型学不会的时候，你会怀疑是模型的问题还是代码有bug，不要犹豫，是代码有bug。

训练模型收敛大概需要2百万步，差不多要24小时+，比较慢，但是很欣慰的是Pong在atari game中是最容易实现的游戏，没有bug的话可以在10小时以内收敛，很良心。

DQN算法非常经典，值得学习，建议大家都自己实现一遍。
