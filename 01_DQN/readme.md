# Pytorch深度强化学习1. Deep Q-Network

我一直对强化学习感兴趣，这学期正好选了一门强化学习的课，第一次作业是让复现DQN。这几年也看了不少DQN的代码，但要自己实现起来，还是犯晕，效率很低。这篇文章从深度强化学习所需的元素出发，达到用DQN解决atari games的目的。代码使用pytorch 1.4版本。

## 1. Observe, Value, Act
强化学习研究的是Agent和环境交互中如何学习最优策略，以获得最大收益。Agent需要能够观察环境(observe)的到所处的状态，评判（value）状态下每个动作的价值，并选出最优的动作（act）来和环境交互。因此，observe，value和act是强化学习Agent必不可少的元素。
```python
class DQNAgent: 

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
```
在atari game中，环境给出的observation（84x84x1的array）可以直接作为state，observe()函数可以帮忙把numpy array转换为torch tensor。在更复杂的partial observable环境，我们可以利用observation来推断所处的state，observe()函数可以更复杂些。value()在DQN中主要是给出state下每个action的Q value。act()则是通过epsilon greedy policy选出action。

## 2. Deep-Q Network
对于复杂的问题，state维度非常大，我们很难基于tabular method来判断每一个(state, action)的价值。这种情况下，我们利用function approximation方法，构建一个深度神经网络(Deep-Q Network, DQN)，来估计(state, action)的价值。value()中Deep-Q Network模块就是一个神经网络，输入是atari game中的一帧图像，输出是每个action的价值。这里利用别人现成的代码（pytorch）。
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)
```
## 3. Learning
强化学习，指的就是Agent与环境交互，不断强化，不断学习，最终找到解决问题的最优策略，那么学习（learn）就是强化学习必不可少的元素。在深度强化学习中，主要关注的就是DQN权重的学习，与深度神经网络的学习很相近，都是算一个loss，然后通过back-prop更新权重。只不过，DQN的loss是temporal difference loss，源自Bellman Equation：

$$Loss_{TD}=R_t+\gamma Q(s_{t+1},a_{t+1})−Q(s_t,a_t)$$

我们把这learn()和TD loss加入Agent中。
```python
class DQNAgent(Agent): 
    def __init__(self):
        self.DQN = DQN(...)

    ...
    
    def compute_td_loss(self, states, actions, rewards, next_states, is_done, gamma=0.99):
        return td_loss

    def learn_from_experience(self, batch_size):
        td_loss = compute_TD_loss(...)
        update_weight(td_loss)
```
## 4. Memory 
因为深度神经网络收敛很慢，需要非常多的样本，如果只根据环境交互来训练网络，将非常的没效率。因此DQN引入了一个memory buffer来进行memory replay，就是把之前和环境交互的经验存下来，在训练时重复利用。memory buffer主要实现两个函数：push将经验存入，sample将经验取出用于训练。
```python
class Memory_Buffer(object):
    def __init__(self, memory_size=1000):
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
## 5. Environment
接下来就是配置环境：atari-games environment，以前windows装这个环境还很麻烦，现在好像直接pip就可以安装，很方便。另外利用baselines包中的wrap_deepmind函数，将环境输出的frame转换成84 x 84的array，方便训练。游戏就选Pong：玩家和电脑每人拿一个板子，接对方弹来的球，没接住的话，对方得一分，先得到21分的获胜。
```python
import gym
from baselines.common.atari_wrappers import make_atari, wrap_deepmind, LazyFrames

# Create and wrap the environment
env = make_atari('PongNoFrameskip-v4') # only use in no frameskip environment
env = wrap_deepmind(env, scale = False, frame_stack=True)
env.reset()
```
​
![](./img/1-1.png)

之后要做的就是补全代码，debug，此处省略24小时。。。

## 6. Results：
​
下面我们看一下训练结果：

![](./img/1-2.png)

上图中，x轴是训练的frame数量，y轴是近10个episodes的平均rewards。蓝色和橘黄色的线分别对应无framestack和有framestack的模型。Frame stack指的是将近期的4个frame贴合到一起作为state，单个frame(84, 84, 1)，组成一起就是(84, 84, 4)。Frame stack可以提供temporal的信息，同时可以加速exploration（原来每步训练，现在每4步训练），也是最早DQN paper中所使用的一个trick，对训练非常有帮助。可以看到，有frame stack的时候模型在40万步的时候，平均收益达到20。

另外一个trick就是target model，即算TD error时候， Q(st+1,at+1) 使用target model来估计，target model和current model每1000步同步一次，保证训练的稳定性。
## 7. Thoughts
感想就是，总结强化学习需要的元素相对容易，真正实现的时候很麻烦，尤其是当模型学不会的时候，你会怀疑是模型的问题还是代码有bug，不要犹豫，是代码有bug。

训练有framestack的模型2百万步，差不多要24小时+，比较慢，但是很欣慰的是Pong在atari game中是最容易实现的游戏，没有bug的话可以在10小时以内收敛，很良心。

代码会以jupyter notebook形式放在github上，有需要的可以去看：https://github.com/AmazingAng/deep-RL-elements