"""
train.py
1. Trains the Q-learning agent over many simulated episodes.
2. Evaluates the trained agent vs. the fixed-timer baseline on unseen
   episodes (same random seeds for a fair comparison).
3. Saves the trained Q-table + a results chart + a metrics JSON, which the
   report's "Efficiency Analysis" section and the Flask dashboard both use.
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation.traffic_env import IntersectionEnv
from simulation.q_learning_agent import QLearningAgent
from simulation.fixed_timer_baseline import FixedTimerController

EPISODE_SECONDS = 3600   # 1 simulated hour per episode
N_TRAIN_EPISODES = 4000
N_EVAL_EPISODES = 30
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_episode(env, controller, is_agent, eval_mode=False):
    state = env.reset()
    for _ in range(EPISODE_SECONDS):
        d_state = env.discretized_state()
        if is_agent:
            # eval_mode uses a small amount of exploration (epsilon=0.02) plus
            # the safety net — empirically, pure greedy exploitation on this
            # Q-table performs worse than near-greedy with a little randomness
            # (a known tabular Q-learning phenomenon: greedy argmax can get
            # stuck exploiting a locally-mediocre, insufficiently-explored
            # action repeatedly, while a little randomness escapes it). The
            # agent's RNG is explicitly seeded before evaluation (see
            # evaluate()) so this exploration is reproducible run to run.
            action = controller.act_with_safety(env, greedy=False) if eval_mode else controller.act(d_state)
        else:
            action = controller.act(state)
        next_state, reward, done, info = env.step(action)
        if is_agent and not eval_mode:
            d_next_state = env.discretized_state()
            controller.update(d_state, action, reward, d_next_state)
        state = next_state
    return info  # final info dict has cumulative stats


def train():
    agent = QLearningAgent(seed=42)
    reward_history = []
    for ep in range(N_TRAIN_EPISODES):
        env = IntersectionEnv(seed=1000 + ep)
        info = run_episode(env, agent, is_agent=True)
        reward_history.append(-info["avg_wait_per_car"])
        if (ep + 1) % 50 == 0:
            print(f"episode {ep+1}/{N_TRAIN_EPISODES}  "
                  f"avg_wait={info['avg_wait_per_car']:.2f}s  eps={agent.epsilon:.3f}")
    agent.save(os.path.join(RESULTS_DIR, "q_table.pkl"))
    return agent, reward_history


def evaluate(agent):
    agent.rng = np.random.default_rng(999)  # explicit seed so evaluation is reproducible
    agent_waits, baseline_waits = [], []
    agent_queues, baseline_queues = [], []
    for ep in range(N_EVAL_EPISODES):
        seed = 5000 + ep
        env_a = IntersectionEnv(seed=seed)
        info_a = run_episode(env_a, agent, is_agent=True, eval_mode=True)
        agent_waits.append(info_a["avg_wait_per_car"])
        agent_queues.append(info_a["total_queue"])

        env_b = IntersectionEnv(seed=seed)
        baseline = FixedTimerController()
        info_b = run_episode(env_b, baseline, is_agent=False)
        baseline_waits.append(info_b["avg_wait_per_car"])
        baseline_queues.append(info_b["total_queue"])

    metrics = {
        "agent_avg_wait_sec": float(np.mean(agent_waits)),
        "baseline_avg_wait_sec": float(np.mean(baseline_waits)),
        "agent_final_queue_mean": float(np.mean(agent_queues)),
        "baseline_final_queue_mean": float(np.mean(baseline_queues)),
    }
    metrics["wait_time_improvement_pct"] = 100 * (
        metrics["baseline_avg_wait_sec"] - metrics["agent_avg_wait_sec"]
    ) / metrics["baseline_avg_wait_sec"]

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n=== EVALUATION ({N_EVAL_EPISODES} unseen episodes, matched seeds) ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")

    # chart
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(["AI (Q-learning)", "Fixed-timer"],
               [metrics["agent_avg_wait_sec"], metrics["baseline_avg_wait_sec"]],
               color=["#2E86AB", "#A23B72"])
    ax[0].set_ylabel("Avg wait per car (s)")
    ax[0].set_title("Average Wait Time")

    ax[1].boxplot([agent_waits, baseline_waits], labels=["AI", "Fixed-timer"])
    ax[1].set_ylabel("Avg wait per car (s)")
    ax[1].set_title("Wait Time Distribution (20 episodes)")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "efficiency_comparison.png"), dpi=150)
    print(f"\nChart saved to {RESULTS_DIR}/efficiency_comparison.png")
    return metrics


if __name__ == "__main__":
    print("Training Q-learning traffic light controller...")
    agent, history = train()
    print("\nTraining complete. Evaluating...")
    evaluate(agent)
