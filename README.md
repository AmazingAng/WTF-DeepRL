# WTF 深度强化学习

用 Python 3.13 创建环境并运行 `python -m pip install -r requirements.txt`。在仓库根目录启动 Notebook 服务，然后打开各章 notebook。课程使用 Gymnasium 的 `ALE/Pong-v5`。单步示例在终止或截断时重置环境，只对终止遮蔽后续价值。N-step 示例将截断也视为窗口边界，避免采样跨越两局。
WTF 深度强化学习算法库及教程，学习并复现经典的深度强化学习算法。

深度强化学习算法使用pytorch实现，写在jupyter notebook中。用Atari Games中的Pong测试。

## 先修课程

1. 机器学习/深度学习

2. python编程

3. 经典强化学习

## 目录

**第1讲：Deep Q-learning**：[文章](https://github.com/AmazingAng/WTF-DeepRL/tree/master/01_DQN/readme.md) | [Code](https://github.com/AmazingAng/WTF-DeepRL/tree/master/01_DQN) | [Google Colab](https://colab.research.google.com/github/AmazingAng/WTF-DeepRL/blob/master/01_DQN/DQN.ipynb#)

**第2讲：Prioritized Experience Replay (PER)**：[代码](https://github.com/AmazingAng/deep-RL-elements/tree/master/02_DQN_PER) | [文章](https://github.com/AmazingAng/deep-RL-elements/tree/master/02_DQN_PER/readme.md)

**第3讲：Double DQN (DDQN)**：[代码](https://github.com/AmazingAng/deep-RL-elements/tree/master/03_DDQN) | [文章](https://github.com/AmazingAng/deep-RL-elements/tree/master/03_DDQN/readme.md)

**第4讲：Dueling DQN**：[代码](https://github.com/AmazingAng/deep-RL-elements/tree/master/04_DuelingDQN) | [文章](https://github.com/AmazingAng/deep-RL-elements/tree/master/04_DuelingDQN/readme.md)

**第5讲：DQN with Soft Update (Soft DQN)**：[代码](https://github.com/AmazingAng/deep-RL-elements/tree/master/05_SoftDQN) | [文章](./05_SoftDQN/readme.md)

**第6讲：DQN with Noisy Net**：[代码](https://github.com/AmazingAng/deep-RL-elements/tree/master/06_NoisyDQN) | [文章](./06_NoisyDQN/readme.md)

**第7讲：DQN with Intrinsic Curiosity Module**：[代码](https://github.com/AmazingAng/deep-RL-elements/tree/master/07_ICMDQN) | [文章](https://github.com/AmazingAng/deep-RL-elements/tree/master/07_ICMDQN/readme.md)

**第8讲：Deep Recurrent Q-learning (DRQN)**：[代码](https://github.com/AmazingAng/deep-RL-elements/tree/master/08_DRQN) | [文章](https://github.com/AmazingAng/deep-RL-elements/tree/master/08_DRQN/readme.md)

**第9讲：N-step DQN**：[代码](https://github.com/AmazingAng/deep-RL-elements/tree/master/09_NStepDQN) | [文章](https://github.com/AmazingAng/deep-RL-elements/tree/master/09_NStepDQN/readme.md)

## Prerequisite 
- torch 1.4.0
- gym[atari]
- tensorboardX 

## 其他深度强化学习算法库 (pytorch)
- [Reinforcement-Learning](https://github.com/andri27-ts/Reinforcement-Learning)
- [DeepRL-Tutorials](https://github.com/qfettes/DeepRL-Tutorials)
- [Deep-Reinforcement-Learning-Algorithms-with-PyTorch](https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch)

