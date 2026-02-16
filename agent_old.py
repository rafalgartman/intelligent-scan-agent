import os
import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from environments import Env


# ----------------------------
# Utilities
# ----------------------------
def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_state(obs, device: torch.device) -> torch.Tensor:
    """
    Convert observation (torch.Tensor / np.ndarray / list) into a detached float32 tensor on `device`.
    Returns shape (obs_dim,) (no batch dimension).
    """
    if isinstance(obs, torch.Tensor):
        return obs.detach().to(device=device, dtype=torch.float32).view(-1)
    if isinstance(obs, np.ndarray):
        return torch.from_numpy(obs).to(device=device, dtype=torch.float32).view(-1)
    return torch.tensor(obs, device=device, dtype=torch.float32).view(-1)


# ----------------------------
# Model
# ----------------------------
class DQN(nn.Module):
    def __init__(self, n_observations: int, n_actions: int):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 16)
        self.layer2 = nn.Linear(16, 32)
        self.layer3 = nn.Linear(32, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, obs) or (obs,)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)  # (B, n_actions)


# ----------------------------
# Agent
# ----------------------------
class Agent:
    def __init__(
        self,
        env,
        exploration_decay_fun,
        num_episodes: int = 20000,
        batch_size: int = 128,
        gamma: float = 0.7,
        tau: float = 0.003,
        lr: float = 1e-5,
        memory_length: int = 10000,
        guess_enabled: bool = False,
        weights_path_best: str = "pretrained_best.pth",
    ):
        self.env = env
        self.exploration_decay = exploration_decay_fun

        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.guess_enabled = guess_enabled

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Infer dimensions
        init_state = env.reset()
        init_state_t = to_state(init_state, self.device)
        n_observations = int(init_state_t.numel())
        n_actions = len(env.actions)

        # Nets
        self.policy_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net = DQN(n_observations, n_actions).to(self.device)
        self.best_net = DQN(n_observations, n_actions).to(self.device)

        # Load pretrained if exists
        self.fname_best = weights_path_best
        if os.path.exists(self.fname_best):
            self.policy_net.load_state_dict(
                torch.load(self.fname_best, map_location=self.device, weights_only=True)
            )
            print(f"Loaded {self.fname_best} as policy net")
        else:
            print("No pretrained policy net found. Initialized random parameters.")

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.best_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.criterion = nn.SmoothL1Loss()

        # Replay memory stores: (state_1d, action_int, reward_float, next_state_1d_or_None, done_bool)
        self.memory = deque([], maxlen=memory_length)

        self.episode = 0
        self.rewards = []

    def select_action(self, state_1d: torch.Tensor) -> int:
        """
        state_1d: shape (obs,)
        returns: Python int action
        """
        rnd = random.random()
        training = self.policy_net.training
        eps = self.exploration_decay(self.episode, self.num_episodes) if training else 0.0

        # Action 5 ("guessing") is off
        if not self.guess_enabled:
            if rnd > eps and self.env.counter[0] < 2 and self.env.counter[1] < 2:
                with torch.no_grad():
                    q = self.policy_net(state_1d.unsqueeze(0))  # (1, n_actions)
                    if training:
                        a = int(torch.argmax(q[0, 0:4]).item())
                    else:
                        a = int(torch.argmax(self.best_net(state_1d.unsqueeze(0))[0, 0:4]).item())
                return a

            self.env.counter[0] = 0
            self.env.counter[1] = 0
            return int(np.random.choice(4))

        # Guessing mode enabled
        if rnd < eps:
            return 4  # guess is always a random action - net is pretrained
        with torch.no_grad():
            q = self.policy_net(state_1d.unsqueeze(0)) if training else self.best_net(state_1d.unsqueeze(0))
            return int(torch.argmax(q[0]).item())

    def optimize_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # (B, obs)
        state_batch = torch.stack(states, dim=0).to(self.device)

        # (B, 1) long indices
        action_batch = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)

        # (B,) float rewards
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        # Q(s,a) -> (B,)
        q = self.policy_net(state_batch)  # (B, n_actions)
        q_sa = q.gather(1, action_batch).squeeze(1)

        # Bootstrap target: 0 for terminals
        non_final_mask = torch.tensor([not d for d in dones], dtype=torch.bool, device=self.device)
        next_state_values = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            if non_final_mask.any():
                non_final_next_states = torch.stack(
                    [ns for ns in next_states if ns is not None], dim=0
                ).to(self.device)
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        target = reward_batch + self.gamma * next_state_values  # (B,)
        loss = self.criterion(q_sa, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # Soft update target net (clean in-place param update)
        with torch.no_grad():
            for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
                tp.mul_(1.0 - self.tau).add_(pp, alpha=self.tau)

    def train(self, save_last_path: str = "pretrained_last.pth", save_best_path: str = "pretrained_best.pth"):
        highest_rwd = -np.inf
        highest_episode = 0
        self.policy_net.train()

        for episode in range(self.num_episodes):
            self.episode = episode

            state = to_state(self.env.reset(), self.device)
            reward_tot = 0.0

            while True:
                action = self.select_action(state)
                observation, reward, done, info_ = self.env.step(action)
                reward_tot += float(reward)

                next_state = None if done else to_state(observation, self.device)

                # Store transition (reward as float)
                self.memory.append((state, action, float(reward), next_state, bool(done)))

                # Optimize on this transition
                self.optimize_step()

                # Only advance state if not terminal
                if not done:
                    state = next_state

                if done:
                    self.rewards.append(reward_tot)
                    if reward_tot > highest_rwd:
                        highest_rwd = reward_tot
                        highest_episode = episode
                        self.best_net.load_state_dict(self.policy_net.state_dict())
                    break

            if episode % 100 == 0:
                last_n = self.rewards[-100:] if len(self.rewards) >= 100 else self.rewards
                mean_last = float(np.mean(last_n)) if last_n else float("nan")
                print(
                    f"Episode: {episode}, Highest reward: {highest_rwd:.3f}, "
                    f"last {len(last_n)} mean: {mean_last:.3f}"
                )

        # Save
        torch.save(self.policy_net.state_dict(), save_last_path)
        torch.save(self.best_net.state_dict(), save_best_path)

        # Plot
        episodes = np.arange(len(self.rewards))
        window = 100
        if len(self.rewards) > window:
            plt.figure()
            plt.plot(highest_episode, highest_rwd, "or", label="max")
            plt.plot(episodes, self.rewards, label="rewards")
            moving_avg = np.convolve(np.array(self.rewards), np.ones(window) / window, mode="valid")
            plt.plot(episodes[window - 1 :], moving_avg, "r", label="moving average")

            ax = plt.twinx()
            ax.plot(episodes, self.exploration_decay(episodes, self.num_episodes), "k", label="epsilon")

            plt.legend()
            plt.tight_layout()
            plt.show()

    def test_walk(self, episodes: int = 3):
        self.policy_net.eval()
        self.best_net.eval()

        for _ in range(episodes):
            state = to_state(self.env.reset(), self.device)
            total_reward = 0.0
            fig, ax = plt.subplots(3)

            while True:
                self.episode = int(1e18)  # effectively epsilon ~ 0
                action = self.select_action(state)
                observation, reward, done, info = self.env.step(action)
                total_reward += float(reward)

                image = self.env.hidden_image[0, 0, :, :]
                masked_image = self.env.masked_image[0, 0, :, :]
                image_mask = self.env.scan_mask[0, 0, :, :]

                ax[0].imshow(image, cmap="gray")
                lbl = f"guess: {int(np.argmax(self.env.predictions_history[-1]))} ({int(self.env.hidden_label)})"
                ax[0].set_title(lbl)
                ax[0].axis("off")

                ax[1].imshow(image_mask, cmap="gray")
                ax[1].axis("off")

                ax[2].imshow(masked_image, cmap="gray")
                lbl = f"step:{self.env.step_size}, bs:{self.env.counter[0]}, rs:{self.env.counter[1]}, rwd:{reward:.1f}"
                ax[2].set_title(lbl)
                ax[2].axis("off")

                plt.tight_layout()
                plt.pause(0.1)

                if done:
                    print(f"total_reward = {total_reward:.3f}")
                    print(f"pixels explored: {int(self.env.scan_mask.sum())}")
                    break

                state = to_state(observation, self.device)

    def test_success_rate(self, n_trials: int = 100):
        self.policy_net.eval()
        self.best_net.eval()

        success = 0.0
        success_full = 0.0
        steps_sum = 0.0

        for _ in range(n_trials):
            state = to_state(self.env.reset(), self.device)
            done = False

            while not done:
                self.episode = int(1e18)  # exploration off
                action = self.select_action(state)
                observation, reward, done, info = self.env.step(action)

                if done:
                    # assuming info[0], info[1] are 0/1
                    success += float(info[0])
                    success_full += float(info[1])
                    steps_sum += float(self.env.step_count)
                    break

                state = to_state(observation, self.device)

        sr = 100.0 * success / n_trials
        sr_full = 100.0 * success_full / n_trials
        mean_steps = steps_sum / n_trials

        print(f"SUCCESS RATE: {sr:.1f}% (full picture accuracy: {sr_full:.1f}%)")
        print(f"MEAN NUMBER OF STEPS: {mean_steps:.1f}")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    set_global_seed(42)

    def exploration_decay(episode, num_episodes):
        return np.exp(-episode / num_episodes * 5)

    env = Env(
        rng=3,
        nsteps=20,
        explore_rwd=0.2,
        bounce_rwd=-1,
        return_rwd=-1,
        hit_rwd=0.2,
        miss_rwd=-0.0,
        rng_discover=1,
        step_rwd=-0.0,
    )

    agent = Agent(
        env,
        exploration_decay_fun=exploration_decay,
        num_episodes=20000,
        batch_size=128,
        gamma=0.7,
        tau=0.003,
        lr=1e-5,
        memory_length=10000,
        guess_enabled=False,
    )

    agent.train()
    agent.test_walk()
    agent.test_success_rate()

    # Second run: guessing enabled
    env2 = Env(
        rng=3,
        nsteps=20,
        explore_rwd=0.1,
        bounce_rwd=-1,
        return_rwd=-1,
        hit_rwd=1,
        miss_rwd=-1,
        rng_discover=1,
        step_rwd=-0.0,
    )

    agent2 = Agent(
        env2,
        exploration_decay_fun=exploration_decay,
        num_episodes=10000,
        batch_size=128,
        gamma=0.95,
        tau=0.003,
        lr=0.5e-5,
        memory_length=10000,
        guess_enabled=True,
    )

    agent2.train()
    agent2.test_walk()
    agent2.test_success_rate()
