"""
q_learning_agent.py
Tabular Q-learning controller for the intersection.

This is the project's AI/ML component: the agent learns, purely from
trial-and-error reward signal (negative total queue length), which phase
to hold at each discretised state. No hand-written if/else signal logic
drives normal operation.

SAFETY OVERRIDE (act_with_safety): the queue-bucket discretisation caps
everything at "16 or more" (bucket 4) — so a queue of 16 and a queue of
40 look identical to the learned policy. Combined with severe queue
imbalance being a rare state during training, the learned Q-values there
can be close to noise, occasionally picking the wrong action exactly in
the worst-case scenario (found via real testing — see
TEST_CASES.md / the conversation this was debugged in). act_with_safety
wraps the learned policy with a rule-based failsafe for this specific
extreme case: if one queue is severely backed up, the other is nearly
empty, and the light has been on long enough to legally switch, force
the switch regardless of what the (undertrained, in this corner) Q-table
says. This is a standard real-world ITS pattern — RL for normal
operation, a rule-based failsafe for extreme/out-of-distribution states.
"""

import numpy as np
import pickle

N_QUEUE_BINS = 5
N_PHASES = 2
N_TIP_BINS = 3
ACTIONS = [0, 1]  # 0 = NS green, 1 = EW green
NS_GREEN, EW_GREEN = 0, 1

SAFETY_QUEUE_THRESHOLD = 20   # "severely backed up"
SAFETY_RATIO = 3               # backed-up side must be at least this many times the other


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

    def act_with_safety(self, env, greedy=True):
        """Use this at real decision points (live dashboard, evaluation) —
        NOT during training, where the agent needs to experience its own
        mistakes to learn from them. Wraps act() with the safety override
        described in the module docstring."""
        base_action = self.act(env.discretized_state(), greedy=greedy)

        can_switch = env.time_in_phase >= 10  # matches MIN_GREEN_SEC in traffic_env.py
        ns_backed_up = (env.queue_ns >= SAFETY_QUEUE_THRESHOLD and
                        env.queue_ns >= SAFETY_RATIO * max(1, env.queue_ew))
        ew_backed_up = (env.queue_ew >= SAFETY_QUEUE_THRESHOLD and
                        env.queue_ew >= SAFETY_RATIO * max(1, env.queue_ns))

        if can_switch:
            if ns_backed_up and env.phase == EW_GREEN:
                return NS_GREEN
            if ew_backed_up and env.phase == NS_GREEN:
                return EW_GREEN
        return base_action

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
