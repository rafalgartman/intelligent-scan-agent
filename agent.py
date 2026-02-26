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
from utils import set_global_seed, to_state, load_or_keep


# ----------------------------
# Model
# ----------------------------

class DQN(nn.Module):
    """
    Simple MLP that outputs Q-values for each action:
    Q(s) -> [Q(s,a0), Q(s,a1), ..., Q(s,aN-1)]
    """
    def __init__(self, n_observations: int, n_actions: int):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 16)
        self.layer2 = nn.Linear(16, 32)
        self.layer3 = nn.Linear(32, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (obs,) or (B, obs). Convenient for single-step and batched training.
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
        fname: str = "data/pretrained_best.pth",
    ):
        """
        DQN Agent with:
        - experience replay (deque memory)
        - target network (stabilizes learning)
        - soft target updates (tau)

        guess_enabled:
          False -> only navigate (actions 0..3)
          True  -> navigate + guess (action 4)
        """
        self.env = env
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.guess_enabled = guess_enabled
        self.exploration_decay = exploration_decay_fun

        # Huber loss is standard for DQN (more robust than MSE to outliers).
        self.criterion = nn.SmoothL1Loss()

        # Infer network input/output sizes from the environment.
        state0 = to_state(env.reset())
        n_observations = int(state0.numel())
        n_actions = len(env.actions)

        # policy_net: trained every step
        # target_net: slow-moving copy used to compute stable TD targets
        # best_net: snapshot of best-performing policy seen so far (by episode reward)
        self.policy_net = DQN(n_observations, n_actions)
        self.target_net = DQN(n_observations, n_actions)
        self.best_net = DQN(n_observations, n_actions)

        self.fname = fname
        if os.path.exists(self.fname):
            # weights_only=True avoids loading arbitrary python objects (safer)
            loaded = load_or_keep(self.policy_net, self.fname)
            self.policy_net.load_state_dict(loaded)
            print(
                f"Policy net weights applied from: {self.fname}" if loaded is not self.policy_net.state_dict() else "Policy net kept (no compatible checkpoint).")
        else:
            print("No pretrained policy net found. Initialized random parameters.")

        # Start target and best aligned with policy.
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.best_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr, amsgrad=True)

        self.episode = 0
        self.rewards = []

        # Replay buffer: stores transitions for off-policy learning
        # store: (state_1d, action_int, reward_float, next_state_1d_or_None, done_bool)
        self.memory = deque([], maxlen=memory_length)

    def select_action(self, state_1d, net=None) -> int:
        """
        Epsilon-greedy action selection:
          - with probability eps: explore (random)
          - otherwise: exploit (argmax Q)

        Note: You explicitly restrict to actions [0..3] when guess_enabled=False.
        """
        net = self.policy_net if net is None else net
        rnd = random.random()
        training = net.training
        eps = self.exploration_decay(self.episode, self.num_episodes) if training else 0.0


        # NAVIGATION ONLY MODE
        if not self.guess_enabled:
            # "counter" looks like your env-specific anti-stuck logic:
            # if counters are low and rnd > eps -> exploit; otherwise explore.
            if rnd > eps and self.env.counter[0] < 2 and self.env.counter[1] < 2:
                with torch.no_grad():
                    q = net(state_1d.unsqueeze(0))[0]  # (n_actions,)
                    # Only consider the 4 movement actions in this phase.
                    return int(torch.argmax(q[0:4]).item())

            # Reset counters when exploring (env-specific behavior).
            self.env.counter[0] = 0
            self.env.counter[1] = 0
            return int(np.random.choice(4))


        # GUESSING ENABLED MODE
        # NOTE: Here you currently explore by selecting action=4 (guess).
        # That is an unusual exploration strategy (normally random among all actions).
        # If it’s intentional (force agent to try guessing early), keep it.
        if rnd < eps and self.env.step_count > 10:  # I want to agent to exprience guessing after 10nth step
            return 4

        with torch.no_grad():
            q = net(state_1d.unsqueeze(0))[0]
            return int(torch.argmax(q).item())

    def optimize_step(self):
        """
        One DQN optimization step using a random minibatch from replay buffer.

        Core idea:
          target = r + gamma * max_a' Q_target(s', a')
          loss = Huber(Q_policy(s, a), target)
        """
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Batch tensors
        state_batch = torch.stack(states, dim=0)  # (B, obs)
        action_batch = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)  # (B, 1)
        reward_batch = torch.tensor(rewards, dtype=torch.float32)  # (B,)

        # Q(s, a) for actions taken
        q = self.policy_net(state_batch)                    # (B, n_actions)
        q_sa = q.gather(1, action_batch).squeeze(1)         # (B,)

        # Next-state bootstrap values; terminals get 0 by definition.
        non_final_mask = torch.tensor([not d for d in dones], dtype=torch.bool)
        next_state_values = torch.zeros(self.batch_size, dtype=torch.float32)

        with torch.no_grad():
            if non_final_mask.any():
                non_final_next_states = torch.stack([ns for ns in next_states if ns is not None], dim=0)
                # max over actions a' of Q_target(s', a')
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        # TD target
        target = reward_batch + self.gamma * next_state_values  # (B,)
        loss = self.criterion(q_sa, target)

        # Backprop + gradient clipping for stability
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # Soft update target network:
        # target = (1-tau)*target + tau*policy
        with torch.no_grad():
            for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
                tp.mul_(1.0 - self.tau).add_(pp, alpha=self.tau)

    def train(self):
        """
        Main training loop:
          for each episode:
            run until done:
              select action -> env.step -> store transition -> optimize_step
            track episode reward and snapshot best_net
        """
        highest_rate = -np.inf

        for episode in range(self.num_episodes):
            self.policy_net.train()
            self.episode = episode

            state = to_state(self.env.reset())
            reward_tot = 0.0

            while True:
                action = self.select_action(state)
                observation, reward, done, info_ = self.env.step(action)

                reward_tot += float(reward)
                next_state = None if done else to_state(observation)

                # Store transition for replay. Using None for terminal next_state is standard.
                self.memory.append((state, action, float(reward), next_state, bool(done)))

                # Learn from a random minibatch (off-policy)
                self.optimize_step()

                if done:
                    self.rewards.append(reward_tot)
                    break

                state = next_state

            # Logging
            if (episode + 1) % 100 == 0 and episode != 0:
                print(f"Episode: {episode + 1} / {self.num_episodes}")
                _, sr, _ = self.test_success_rate(500)

                # Keep best policy snapshot by total episode reward
                if sr > highest_rate:
                    highest_rate = sr
                    self.best_net.load_state_dict(self.policy_net.state_dict())
                    torch.save(self.best_net.state_dict(), self.fname)

        # Plot training rewards + moving average + epsilon schedule
        episodes = np.arange(len(self.rewards))
        window = 100
        if len(self.rewards) > window:
            plt.figure()
            plt.plot(episodes, self.rewards, label="rewards")
            moving_avg = np.convolve(np.array(self.rewards), np.ones(window) / window, mode="valid")
            plt.plot(episodes[window - 1 :], moving_avg, "r", label="moving average")
            plt.xlabel("episode")
            plt.ylabel("rewards")
            plt.legend()
            ax = plt.twinx()
            ax.plot(episodes, self.exploration_decay(episodes, self.num_episodes), "k", label = "epsilon")
            ax.set_ylabel("epsilon")
            plt.tight_layout()
            plt.show()


    def test_walk(self, n_runs: int = 1):
        """
        Visual debug: run the agent in the env and show:
          - true image
          - scan mask (where agent looked)
          - masked image (what agent has revealed)
        """
        self.policy_net.eval()
        self.best_net.eval()

        for _ in range(n_runs):
            state = to_state(self.env.reset())
            total_reward = 0.0
            _, ax = plt.subplots(3)

            while True:
                # Turn off exploration by forcing episode huge -> eps ~ 0
                self.episode = int(1e18)
                action = self.select_action(state, net=self.best_net)
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

                state = to_state(observation)

    def test_success_rate(self, n_trials: int = 100):
        """
        Run multiple evaluation episodes and report:
          - success rate from partial scan
          - success rate when using full picture (from env info)
          - mean number of steps taken
        """
        self.policy_net.eval()
        self.best_net.eval()

        success = 0.0
        success_full = 0.0
        steps_sum = 0.0
        for _ in range(n_trials):
            state = to_state(self.env.reset())
            done = False

            while not done:
                self.episode = int(1e18)  # exploration off
                action = self.select_action(state, net=self.best_net)
                observation, reward, done, info = self.env.step(action)
                if done:
                    # info expected to contain [partial_success, full_image_success]
                    success += float(info[0]) # 0/1 if correct/ incorrect answer for discovered picture
                    success_full += float(info[1]) # 0/1 if correct/ incorrect answer for full picture
                    steps_sum += float(self.env.step_count)
                    break

                state = to_state(observation)

        success_rate = 100.0 * success / n_trials
        success_rat_full = 100.0 * success_full / n_trials
        mean_steps = steps_sum / n_trials

        print(f"SUCCESS RATE: {success_rate:.1f}% (full picture accuracy: {success_rat_full:.1f}%)")
        print(f"MEAN NUMBER OF STEPS: {mean_steps:.1f}")

        return mean_steps, success_rate, success_rat_full


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    set_global_seed(42)

    # STEP 1: train and save perceptron (used as part of the observation/state),
    # perceptron is loaded by environment to the evionment
    from perceptron_training import train_perceptron
    mdl = train_perceptron(epochs=1, random_masking=False, saving=True)

    # STEP 2: pretrain agent to search (no guessing action)
    # Epsilon decays from 1 -> 0 over training
    def exploration_decay(episode, num_episodes):
        return np.exp(-episode / num_episodes * 3)

    env = Env(
        rng=5,
        nsteps=30,
        explore_rwd=1,
        bounce_rwd=-1,
        return_rwd=-1,
        hit_rwd=0.1,
        miss_rwd=-0.0,
        rng_discover=1,
        step_rwd=-0.0,
    )

    agent = Agent(
        env,
        exploration_decay_fun=exploration_decay,
        num_episodes=1000,
        batch_size=128,
        gamma=0.75,
        tau=0.003,
        lr=2e-4,
        memory_length=50000,
        guess_enabled=False,
    )

    agent.train()
    agent.test_walk()          # visualizes the scan process

    budgets = [5, 10, 15, 20, 25, 30]
    srs, srfs = [], []

    for b in budgets:
        env.nsteps = b
        agent.env = env
        _, sr, srf = agent.test_success_rate()  # change function to return only these two
        srs.append(sr)
        srfs.append(srf)

    plt.figure()
    plt.plot(budgets, srs, marker="o", linestyle="--", label="partial observation")
    plt.plot(budgets, srfs, marker="x", linestyle="--", label="full-image baseline")
    plt.legend()
    plt.xlabel("number of steps")
    plt.ylabel("accuracy")
    plt.show()



    # STEP 3: second training stage with guessing enabled
    env2 = Env(
        rng=5,
        nsteps=30,
        explore_rwd=0.1,
        bounce_rwd=-1,
        return_rwd=-1,
        hit_rwd=5,
        miss_rwd=0,
        rng_discover=1,
        step_rwd=-0.1,
    )

    agent2 = Agent(
        env2,
        exploration_decay_fun=exploration_decay,
        num_episodes=1000,
        batch_size=128,
        gamma=0.75,
        tau=0.003,
        lr=2e-4,
        memory_length=50000,
        guess_enabled=True,
    )

    agent2.train()
    agent2.test_walk()
    agent2.test_success_rate()
