import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import collections

# hyper-parameters
EPISODES = 10000  # 训练/测试幕数
BATCH_SIZE = 64
LR = 0.00025
GAMMA = 0.98
SAVING_IETRATION = 1000  # 保存Checkpoint的间隔
MEMORY_CAPACITY = 5000  # Memory的容量
MIN_CAPACITY = 500  # 开始学习的下限
Q_NETWORK_ITERATION = 10  # 同步target network的间隔
EPSILON = 0.01  # epsilon-greedy
SEED = 0
MODEL_PATH = ''
SAVE_PATH_PREFIX = './log/dqn/'
TEST = False

env = gym.make('CartPole-v1', render_mode="human" if TEST else None)
#env = gym.make('MountainCar-v0', render_mode="human" if TEST else None)
# env = gym.make("LunarLander-v2",continuous=False,gravity=-10.0,enable_wind=True,wind_power=15.0,turbulence_power=1.5,render_mode="human" if TEST else None)


random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(f"{SAVE_PATH_PREFIX}/ckpt", exist_ok=True)

NUM_ACTIONS = env.action_space.n  # 2
NUM_STATES = env.observation_space.shape[0]  # 4
ENV_A_SHAPE = 0 if np.issubdtype(type(env.action_space.sample()),
                                 int) else env.action_space.sample().shape  # 0, to confirm the shape


class Model(nn.Module):
    def __init__(self, num_inputs=4):
        super(Model, self).__init__()

        self.linear = nn.Linear(NUM_STATES, 64)
        self.linear2 = nn.Linear(64, 256)
        self.linear3 = nn.Linear(256, NUM_ACTIONS)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        x = F.relu(self.linear2(x))
        return self.linear3(x)


class Data:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


class Memory:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def set(self, data):
        self.buffer.append(data)

    def get(self, batch_size):
        minibatch = random.sample(self.buffer, batch_size)
        states = np.array([data.state for data in minibatch])
        actions = np.array([data.action for data in minibatch])
        rewards = np.array([data.reward for data in minibatch])
        next_states = np.array([data.next_state for data in minibatch])
        dones = np.array([data.done for data in minibatch])
        return states, actions, rewards, next_states, dones



class DQN():
    """docstring for DQN"""

    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Model().to(device), Model().to(device)
        self.learn_step_counter = 0
        self.memory1_counter = 0
        self.memory1 = Memory(capacity=MEMORY_CAPACITY)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state, EPSILON=1.0):
        state = torch.tensor(state, dtype=torch.float).to(device)
        if np.random.random() > EPSILON:  # random number
            # greedy policy
            action_value = self.eval_net.forward(state)
            action = torch.argmax(action_value).item()
            action = action
            if ENV_A_SHAPE != 0 and isinstance(action, np.ndarray):
                action = action.reshape(ENV_A_SHAPE)
        else:
            # random policy
            action = np.random.randint(0, NUM_ACTIONS)  # int random number
            action = action
            if ENV_A_SHAPE != 0 and isinstance(action, np.ndarray):
                action = action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, data):
        self.memory1.set(data)
        self.memory1_counter += 1

    def learn(self):
        # update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        if self.learn_step_counter % SAVING_IETRATION == 0:
            self.save_train_model(self.learn_step_counter)

        self.learn_step_counter += 1

        # TODO

        # Sample a batch of experiences from memory
        states, actions, rewards, next_states, dones = self.memory1.get(BATCH_SIZE)

        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(device)
        dones = torch.tensor(dones, dtype=torch.float).to(device)

        # Compute Q values for current states
        q_eval = self.eval_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute Q values for next states
        q_next = self.target_net(next_states).detach()
        #q_target = rewards + GAMMA * q_next.max(1)[0] * (1 - dones)
        q_target = rewards + GAMMA * q_next.max(1)[0] * (1 - dones)

        # Compute loss
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_train_model(self, epoch):
        torch.save(self.eval_net.state_dict(), f"{SAVE_PATH_PREFIX}ckpt/{epoch}.pth")

    def load_net(self, file):
        self.eval_net.load_state_dict(torch.load(file))
        self.target_net.load_state_dict(torch.load(file))


def main():
    dqn = DQN()

    writer = SummaryWriter(f'{SAVE_PATH_PREFIX}')

    if TEST:
        dqn.load_net(MODEL_PATH)
    for i in range(EPISODES):
        print("EPISODE: ", i)
        state, info = env.reset(seed=SEED)

        ep_reward = 0
        while True:
            action = dqn.choose_action(state=state, EPSILON=EPSILON if not TEST else 0)  # choose best action
            next_state, reward, done, truncated, info = env.step(action)  # observe next state and reward
            dqn.store_transition(Data(state, action, reward, next_state, done))
            ep_reward += reward
            if TEST:
                env.render()
            if dqn.memory_counter >= MIN_CAPACITY and not TEST:
                dqn.learn()
                if done:
                    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
            if done:
                if TEST:
                    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
                break
            state = next_state
        writer.add_scalar('reward', ep_reward, global_step=i)


if __name__ == '__main__':
    main()