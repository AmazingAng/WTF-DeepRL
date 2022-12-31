# Pytorch深度强化学习3. Double DQN

## 1. Maximization Bias of Q-learning
不论是深度强化学习的DQN还是传统的Q learning，都有maximization bias，会高估Q value。为什么呢？我们可以看下Q learning更新Q值时的公式：
$$Q(S_t, A_t)=Q(S_t, A_t) + \alpha [R_{t+1}+\gamma \max_aQ(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$$
可以想像，在均衡时，有
$$E(Q(S_t, A_t))=R_{t+1}+\gamma E(\max_aQ(S_{t+1}, A_{t+1}))\geq R_{t+1}+\gamma \max_aE(Q(S_{t+1}, A_{t+1}))$$
也就是说，由于在bootstrap更新的时候，选了最大化Q值的action，同时用这个最大的Q值来做更新，导致了Q值的高估。如果高估了非最优动作的价值，会影响学习效果。

在2010年Hasselt et. al.提出了Double Q learning来解决这一问题。Double Q learning就是学习两套Q值，使得选择最佳动作和估计最佳动作的价值可以在不同的网络上进行。公式如下：
$$Q(S_t, A_t)=Q(S_t, A_t) + \alpha [R_{t+1}+\\\gamma Q_2(S_{t+1},\arg\max_aQ(S_{t+1}, A_{t+1})) - Q(S_t, A_t)]$$
然后Q和Q2分别更新，减轻Maximization Bias。

## 2. Double DQN
Double DQN其实就是Double Q learning在DQN上的拓展。用上面的两套Q值，分别对应DQN的policy network（更新的快）和target network（每隔一段时间与policy network同步）。Double Q learning error如下：
$$Y^{DoubleQ}_t ≡ R_{t+1 }+ γQ(S_{t+1}, \arg\max_a{Q(S_{t+1}, a; \theta_t); θ_t^-} )$$
实现起来也很简单，只需要在DQN计算TD error的时候稍作改动：
```python
def compute_td_loss(self, states, actions, rewards, next_states, is_done, gamma=0.99):
    """ Compute td loss using torch operations only. Use the formula above. """
    actions = torch.tensor(actions).long()    # shape: [batch_size]
    rewards = torch.tensor(rewards, dtype =torch.float)  # shape: [batch_size]
    is_done = torch.tensor(done).bool()  # shape: [batch_size]

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
    ## Where DDQN is different from DQN
    predicted_next_qvalues_current = self.DQN(next_states)
    predicted_next_qvalues_target = self.DQN_target(next_states)
    # compute V*(next_states) using predicted next q-values
    next_state_values =  predicted_next_qvalues_target.gather(1, torch.max(predicted_next_qvalues_current, 1)[1].unsqueeze(1)).squeeze(1)

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
```

## 3. Results
我们来看下训练结果，下图是DDQN（蓝色）和DQN（橘黄）在Pong上面得表现。可以看到DDQN大约比DQN得收敛快10%。

![DDQN（蓝色），Dueling DQN（粉红）和DQN（橘黄）的平均奖励
](./img/3-1.png)

## 4. Thoughts
DDQN的实现很容易，在Pong这个游戏上的提升也不大。可能效果要在更难的任务中才能看出来。

代码在 https://github.com/AmazingAng/deep-RL-elements
