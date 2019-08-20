import math
import random
import numpy as np
import matplotlib
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

global p
p=0
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
Stata = namedtuple('Transition','state')


class ReplayMemory(object):

    def __init__(self, capacity):
           self.capacity = capacity
           self.memory = []
           self.position = 0

    def push(self, *args):
       """Saves a transition."""
       if len(self.memory) < self.capacity:
           self.memory.append(None)
       self.memory[self.position] = Transition(*args)
       self.position = (self.position + 1) % self.capacity

    def rand_sample(self, batch_size):
       return random.sample(self.memory, batch_size)

    def sample(self, batch):
        ### is position really changes each loop?
        return self.memory[self.position-batch-1:self.position-1]
    def __len__(self):
       return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_actions):
        super(DQN, self).__init__()
        global p
        self.linear1 = nn.Sequential(
            nn.Linear(3, 1400),  # linear layer 1
            nn.ELU(),
            nn.Dropout(p),
            #nn.BatchNorm1d(1400)
            )
        self.linear2 = nn.Sequential(
            nn.Linear(1400, 700),  # linear layer 2
            nn.ELU(),
            nn.Dropout(p),
            #nn.BatchNorm1d(700)
            )
        self.linear3 = nn.Sequential(
            nn.Linear(700, 150),  # linear layer 3
            nn.ELU(),
            nn.Dropout(p),
            #nn.BatchNorm1d(150)
            )
        self.linear4 = nn.Sequential(
            nn.Linear(150, n_actions),  # linear layer 4
            nn.ELU(),
            nn.Dropout(p),
            # nn.BatchNorm1d(n_actions)
        )

        #linear_input_size = 3 # input size is 3: Wind Speed, Power(Do i need?) and Break Current
        #self.head = nn.Linear(linear_input_size, outputs)



    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        return x.view(x.size(0),-1)


BATCH_SIZE = 1
GAMMA = 0.3
EPS_START = 1
EPS_END = 0.01
EPS_DECAY = 1000
TARGET_UPDATE = 2

# number of actions the net can choose from (+/- 0.001 amps or nothing)
n_actions_small = 3
n_actions= 3
global expected_reward; global counter_DQN
expected_reward = [0];counter_DQN = [0]
DQN(n_actions).double().to(device)
policy_net = DQN(n_actions).to(device)
target_net = DQN(n_actions).to(device)
policy_net.double()
target_net.double()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
policy_net.zero_grad()
target_net.zero_grad()
global learning_rate
learning_rate=0.00001

optimizer = optim.Adam(policy_net.parameters(),lr=learning_rate)
memory = ReplayMemory(10000)
steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * (steps_done) / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            global p
            p=0
            return policy_net(state).max(1)[1]
    else:
        return torch.tensor([random.randrange(n_actions_small)], device=device, dtype=torch.long)


def optimize_model(step_num):
    sample_size= BATCH_SIZE
    if TARGET_UPDATE < step_num:
        transitions = memory.sample(sample_size)
    else:
        transitions = memory.sample(sample_size)

    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Dropout of 0.5
    global p
    p = 0.5
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.uint8)
    #non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    non_final_next_states = torch.cat(batch.next_state)
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch)
    B=torch.zeros(state_action_values.size(0))
    counteri=0
    for i in B:
        B[counteri]=state_action_values[counteri,action_batch[counteri]]
        counteri += 1

    state_action_values=torch.tensor(B,device=device,dtype=torch.double)
    #state_action_values=torch.gather(state_action_values,-1, torch.unsqueeze(action_batch,1))

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    #=target_net(non_final_next_states).max(0)
    next_state_values=torch.zeros(sample_size, device=device,dtype=torch.double)
    A=target_net(non_final_next_states)
    A2=A.max(1)[0]
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values*GAMMA) + reward_batch
    expected_state_action_values=expected_state_action_values.unsqueeze(1)
    expected_state_action_values.long
    global expected_reward; global counter_DQN
    expected_reward.append(expected_state_action_values.item())
    counter_DQN.append(step_num)
    if step_num % 8000 == 0:
        plt.plot(counter_DQN, expected_reward)
    next_state_values.double()
    # Compute Huber loss

    loss = nn.SmoothL1Loss().cuda()
    A=state_action_values.unsqueeze(1)
    state_action_loss = loss(state_action_values.unsqueeze(1), expected_state_action_values)
    # Optimize the model
    global learning_rate
    if step_num> TARGET_UPDATE*50:
        learning_rate=0.0001
    else:
        learning_rate = 0.0001
    optimizer.zero_grad()
    state_action_loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    print('loss:',state_action_loss.item())

def Learn(state,action_preformed, state_last, reward, step_num):
    state_last = torch.as_tensor(np.array(state_last), device=device)
    action_preformed = torch.as_tensor(np.array([action_preformed]), device=device)
    state = torch.as_tensor(np.array(state), device=device)
    reward = torch.as_tensor(np.array([reward]), device=device)
    ##### probably will be better to devide to two loops - reward and action #####
    memory.push(state_last,action_preformed, state, reward)

    # Select an action to preform next step
    action = select_action(state)

    if step_num%TARGET_UPDATE ==0:
        # now the target net = policy net,
        # but the update of the target net should be every few cycles
        optimize_model(step_num)
        # Perform one step of the optimization (on the target network)

        # Update the target network, copying all weights and biases in DQN
        target_net.load_state_dict(policy_net.state_dict())
    return (action)
