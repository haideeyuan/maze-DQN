import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 10   # target update frequency
MEMORY_CAPACITY = 10
N_ACTIONS = 4
N_STATES = 1
ENV_A_SHAPE = 0

# zyy: ENV Parameters
SCENE = np.array([[0,0,1,1],[0,0,0,1],[1,1,0,0],[1,1,0,0]]) # maze: 0 - road //  1 - wall
ROW, COL = SCENE.shape


SCENE = [j for i in SCENE for j in i] # define maze


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x): # x is state
        x = torch.unsqueeze(torch.FloatTensor([x]), 0)

        # zyy: generate action space
        action_space = self.gen_action_space(torch.tensor(x[0], dtype=torch.int32))

        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)[0]
            # zyy: find max val action in action_space
            # print("action val:", actions_value, "       action space:", action_space)
            value = actions_value[0]
            action = action_space[0]
            for i in action_space:
                if value < actions_value[i]:
                    value = actions_value[i]
                    action = i
        else:   # random
            action = random.choice(action_space)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    def test_actione(self, a, _s):
        s = _s.clone()
        if a == 0:
            s -= COL
        elif a == 1:
            s += ROW
        elif a == 2:
            s -= 1
        elif a == 3:
            s += 1
        return s

    def gen_action_space(self, _s):
        x = _s.clone()
        action_space = list()
        if x >= COL and SCENE[self.test_actione(0, x)] == 0:
            action_space.append(0)
        if x < COL*(ROW-1) and SCENE[self.test_actione(1, x)] == 0:
            action_space.append(1)
        if x % COL != 0 and SCENE[self.test_actione(2, x)] == 0:
            action_space.append(2)
        if (x+1) % COL != 0 and SCENE[self.test_actione(3, x)] == 0:
            action_space.append(3)
        return action_space


class ENV(object):
    def __init__(self, ):
        super(ENV, self).__init__()
        self.s = 0
        self.done = 0
    def init_state(self, s0=0):
        self.s = s0
        return s0
    def take_actione(self, a):
        if a == 0:
            self.s -= COL
        elif a == 1:
            self.s += ROW
        elif a == 2:
            self.s -= 1
        elif a == 3:
            self.s += 1

        done = 0
        r = -1
        if self.s == COL * ROW - 1:
            done = 1
            r = 15

        return self.s, r, done

dqn = DQN()
env = ENV()
print('\nCollecting experience...')
for i_episode in range(400):
    s = env.init_state(0)
    ep_r = 0
    while True:
        a = dqn.choose_action(s)

        # take action: update s_, r, done according to a
        s_, r, done = env.take_actione(a)
        print(">>>>>>>>> take_action >>>>>>>>>>>",s, a, s_, done)
        # exit(0)

        dqn.store_transition(s, a, r, s_)

        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))

        if done:
            break
        s = s_
