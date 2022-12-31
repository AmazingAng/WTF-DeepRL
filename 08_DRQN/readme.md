## Pytorch深度强化学习8. Deep Recurrent Q-Network

1. Why RNN
​

编辑

切换为居中
添加图片注释，不超过 140 字（可选）
一般深度学习利用的神经网络是前馈神经网络，其中的神经元接收上一层的输入，并将计算后的值输出给下一层。循环神经网络（RNN）的神经元多了一个同层的连接（上图中的W），即它们也接受同层神经元的输入。RNN有它独特的优势：新的输出受上一次状态（hidden state）的影响，使得RNN具有短期记忆。
​

编辑
添加图片注释，不超过 140 字（可选）
在很多的强化学习任务中，尤其是部分可见（partially observable）的任务，短期记忆是必须的。举一个简单的例子：在上图GridWorld环境中，agent只能看见自己附近的9个格子。对于agent来说，格子（1，2）和（1，3）的观察值是完全相同的，只有记住之前走过的路线才能知道当前的格子是哪个。DQN通过frame stack技术（把过去4帧叠在一起作为一个输入）巧妙的引入了短期记忆，但是记忆仅限于过去的4帧。如果需要更长的记忆，我们需要将RNN引入DQN。
2. DRQN
DRQN于2015被Hausknecht和Stone提出，本质上是把DQN其中的一个linear layer变成了RNNlayer。由于RNN的加入，DRQN具有短期记忆，不需要frame stack技术也可以在Atari Games中与DQN相似的分数。
3. Code
DRQN的难点主要是要将DQN单帧的记忆改为多帧连续记忆，比如8帧。原先从DQN的replay buffer抽取的batch形状为[N_batch, H, W]，DRQN抽取的batch形状为[N_batch, N_seq, H, W]。我在改造代码的时候bug频出，但只要注意batch的形状，编程难度并不大。下面我介绍DRQN网络和recurrent replay buffer的代码，其余代码附在文章末尾的链接中。
我的DRQN网络实现和原文有所区别，目的是尽可能和DQN网络保持一致，只将DQN网络的最后一层linear layer换成了GRU layer。代码如下：
class DRQN(nn.Module):
    def __init__(self, in_channels=4, num_actions=5, device = torch.device("cpu")):
        """
        Initialize a deep Q-learning network as described in
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
            device: cpu or gpu (cuda:0)
        """
        super(DRQN, self).__init__()
        self.device = device
        self.num_actions = num_actions
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.gru = nn.GRU(512, num_actions, batch_first=True) # input shape (batch, seq, feature)

    def forward(self, x, hidden = None, max_seq = 1, batch_size=1):
        # DQN input B*C*feature (32 4 84 84)
        # DRQN input B*C*feature (32*seq_len 4 84 84)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.reshape(x.size(0), -1)))
        hidden = self.init_hidden(batch_size) if hidden is None else hidden
        # before go to RNN, reshape the input to (barch, seq, feature)
        x = x.reshape(batch_size, max_seq, 512)
        return self.gru(x, hidden)

    def init_hidden(self, batch_size):
        # initialize hidden state to 0
        return torch.zeros(1, batch_size, self.num_actions, device= self.device, dtype=torch.float)

Recurrent Memory Buffer主要实现存储experience （push），抽样一个batch的episodic memory （sample），代码如下：
class Recurrent_Memory_Buffer(object):
    # memory buffer to store episodic memory
    def __init__(self, memory_size=1000, max_seq = 10):
        self.buffer = []
        self.memory_size = memory_size
        self.max_seq = max_seq
        self.next_idx = 0
        
    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if len(self.buffer) <= self.memory_size: # buffer not full
            self.buffer.append(data)
        else: # buffer is full
            self.buffer[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % self.memory_size

    def sample(self, batch_size):
        # sample episodic memory
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in range(batch_size):
            finish = random.randint(self.max_seq, self.size() - 1)
            begin = finish-self.max_seq
            
            data = self.rec_memory_buffer.buffer[begin:finish]
            state, action, reward, next_state, done= zip(*data)
            states.append(np.concatenate([self.observe(state_i) for state_i in state]))
            actions.append(action)
            rewards.append(reward)
            next_states.append(np.concatenate([self.observe(state_i) for state_i in next_state]))
            dones.append(done)

        return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones
    
    def size(self):
        return len(self.buffer)

4. Results
我在Atari Games的Pong上面跑了2百万帧，用时30小时，DRQN算法在80万帧左右收敛，比起DQN要慢一些。但我相信在可见信息更少的任务上，DRQN的表现会好于DQN。
​

编辑

切换为居中
添加图片注释，不超过 140 字（可选）

代码： AmazingAng/deep-RL-elements.com/AmazingAng/deep-RL-elements/blob/master/8_DRQN.ipynb
论文： Deep Recurrent Q-Learning for Partially Observable MDPs
