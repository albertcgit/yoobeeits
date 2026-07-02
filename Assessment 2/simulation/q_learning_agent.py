"""
q_learning_agent.py
Tabular Q-learning controller for the intersection.

This is the project's AI/ML component: the agent learns, purely from
trial-and-error reward signal (negative total queue length), which phase
to hold at each discretised state. No hand-written if/else signal logic.
"""

import numpy as np
import pickle

N_QUEUE_BINS = 5
N_PHASES = 2
N_TIP_BINS = 3
ACTIONS = [0, 1]  # 0 = NS green, 1 = EW green


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.4, epsilon_decay=0.9992, epsilon_min=0.02, seed=None):
        self.q = np.zeros((N_QUEUE_BINS, N_QUEUE_BINS, N_PHASES, N_TIP_BINS, len(ACTIONS)))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.rng = np.random.default_rng(seed)

    def act(self, state, greedy=False):
        if (not greedy) and self.rng.random() < self.epsilon:
            return int(self.rng.choice(ACTIONS))
        return int(np.argmax(self.q[state]))

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.q[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q[state][action]
        self.q[state][action] += self.alpha * td_error
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.q, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.q = pickle.load(f)
        self.epsilon = self.epsilon_min  # act greedily once loaded
