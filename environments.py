import random
from typing import Tuple, Optional

import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

from perceptron_training import Perceptron



# Helpers
def small_pic(
    x: int,
    y: int,
    msk: torch.Tensor,
    scn: torch.Tensor,
    rng: int = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract a local (2*rng+1)x(2*rng+1) patch centered at (x,y).

    Why:
      - The agent only gets a *local* view of the image around its cursor.
      - We clamp (x,y) so the patch is always fully inside the image bounds.

    Inputs:
      msk, scn: tensors shaped (1,1,H,W)
        - msk: scan mask (what has been discovered)
        - scn: masked/visible image (what the agent can currently see)

    Returns:
      (mask_patch, scan_patch) as tensors shaped (2*rng+1, 2*rng+1)
    """
    lgth = msk.shape[-1]  # assumes square images (MNIST 28x28)
    x0 = max(int(x), rng)
    y0 = max(int(y), rng)
    x0 = min(x0, lgth - rng - 1)
    y0 = min(y0, lgth - rng - 1)
    return (
        msk[0, 0, x0 - rng: x0 + rng + 1, y0 - rng: y0 + rng + 1],
        scn[0, 0, x0 - rng: x0 + rng + 1, y0 - rng: y0 + rng + 1],
    )


def load_nn() -> Perceptron:

    model = Perceptron()
    # weights_only=True is safer (avoids unpickling arbitrary objects)
    model.load_state_dict(torch.load("trained_LeNet.pth", weights_only=True, map_location="cpu"))
    model.eval()
    return model


# --------------------------------------------------------
# Envronment
# --------------------------------------------------------
class Env:
    """
    Cursor-scanning environment for active perception.

    Core objects:
      - hidden_image: full ground-truth image (agent never observes directly)
      - scan_mask: binary mask of discovered pixels (1=seen, 0=unseen)
      - masked_image: what the agent can see: hidden_image*scan_mask + empty_value elsewhere

    Observation/state vector layout:
      [x_norm, y_norm]                               (2 floats)
      + flattened local patch around cursor          ((2*rng+1)^2 floats)
      + normalized perceptron logits                 (n_logits floats)

    Actions:
      0 up, 1 down, 2 left, 3 right, 4 guess

    Notable improvements in this version:
      - Dataset loaded once in __init__ (much faster than loading each step/reset).
      - Pure-torch "returned" detection (no numpy round trips).
      - Logit normalization uses clamp_min to avoid std==0 -> NaNs.
      - Returned state is clone() to avoid replay-buffer aliasing bugs.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (28, 28),
        rng: int = 2,                 # local patch half-width for observation
        nsteps: int = 100,            # max steps per episode
        explore_rwd: float = 1.0,     # reward scaling for revealing new pixels/info
        bounce_rwd: float = -1.0,     # penalty for trying to move out of bounds
        return_rwd: float = -1.0,     # penalty for moving but revealing nothing new
        hit_rwd: float = 10.0,        # reward for correct guess
        miss_rwd: float = -10.0,      # penalty for incorrect guess
        trigger_guess: bool = False,  # currently unused; kept for compatibility
        rng_discover: int = 0,        # discovery radius: 0=only current pixel; >0 reveals square around cursor
        step_rwd: float = 0.0,        # per-step penalty (encourages shorter scans)
        verbose: bool = False,
    ):
        self.actions = {"up": 0, "down": 1, "left": 2, "right": 3, "guess": 4}
        self.verbose = verbose

        # First part of state: x_norm, y_norm
        self.nps = 2

        self.size_x, self.size_y = image_size
        self.len_x = self.size_x - 1
        self.len_y = self.size_y - 1

        self.nsteps = int(nsteps)
        self.rng = int(rng)
        self.rng_dis = int(rng_discover)
        self.trigger = trigger_guess

        self.step_rwd = float(step_rwd)

        # Reward parameters
        self.rwd_explore = float(explore_rwd)
        self.rwd_bounce = float(bounce_rwd)
        self.rwd_return = float(return_rwd)
        self.rwd_hit = float(hit_rwd)
        self.rwd_miss = float(miss_rwd)


        # Load dataset
        self.dataset = MNIST(root="./data", train=True, download=True, transform=ToTensor())


        # Load pretrained classifier used in observation
        self.perceptron = load_nn()
        self.perceptron.eval()

        # Image tensors (kept batched: (1,1,H,W) to match perceptron input)
        self.hidden_image = torch.zeros(1, 1, self.size_x, self.size_y)
        self.masked_image = torch.zeros(1, 1, self.size_x, self.size_y)
        self.scan_mask = torch.zeros(1, 1, self.size_x, self.size_y)

        # Unseen pixels are set to a constant; value slightly below 0 can be helpful
        # to distinguish "unknown" from "background black".
        self.empty_image = torch.full((1, 1, self.size_x, self.size_y), -0.1)

        self.hidden_label: Optional[int] = None

        # Determine number of output logits/classes (10 for MNIST, 10 for FashionMNIST)
        with torch.no_grad():
            n_logits = int(self.perceptron(self.empty_image).shape[-1])

        self.patch_len = (2 * self.rng + 1) ** 2
        self.state_len = self.nps + self.patch_len + n_logits
        self.current_state = torch.zeros(self.state_len, dtype=torch.float32)

        # Episode bookkeeping
        self.x = 0
        self.y = 0
        self.step_count = 0

        # Useful for debugging/plots: raw logits over time
        self.predictions_history = []

        # Counters: bounce, return, all_bounce, all_return
        # (you use these in agent logic to detect being stuck)
        self.counter = [0, 0, 0, 0]
        self.step_size = 1  # reserved; could be used for variable stride


    def sample_image(self) -> Tuple[torch.Tensor, int]:
        """
        Sample a random item from the dataset.
        Returns:
          image: (1,1,H,W)
          label: int
        """
        idx = random.randint(0, len(self.dataset) - 1)
        image, label = self.dataset[idx]  # image: (1,H,W)
        return image.unsqueeze(0), int(label)  # (1,1,H,W), int

    def _write_state_patch(self):
        """
        Fill the patch part of the state vector with local visible pixels.
        This is the "local observation" the agent navigates with.
        """
        _, img_patch = small_pic(self.x, self.y, self.scan_mask, self.masked_image, self.rng)
        self.current_state[self.nps: self.nps + self.patch_len] = img_patch.flatten()

    def _write_state_logits(self):
        """
        Fill the last part of the state with normalized perceptron logits computed
        from currently discovered pixels only (hidden_image * scan_mask).

        Normalization matters: raw logits can have varying scale; z-scoring tends
        to help DQN learning stability.
        """
        with torch.no_grad():
            logits = self.perceptron(self.hidden_image * self.scan_mask)[0].detach()  # (n_logits,)
        mean = logits.mean()
        std = logits.std().clamp_min(1e-8)  # avoid division by zero
        logits_norm = (logits - mean) / std

        start = self.nps + self.patch_len
        self.current_state[start:] = logits_norm

        # store raw logits for debug/visualization
        self.predictions_history.append(logits.cpu().numpy())

    def reset(self) -> torch.Tensor:
        """
        Start a new episode:
          - random start position
          - new hidden image + label
          - reset scan_mask and reveal initial region
          - build initial observation vector
        """
        self.x = random.randint(0, self.len_x)
        self.y = random.randint(0, self.len_y)

        self.hidden_image, self.hidden_label = self.sample_image()

        # Reset scan mask to all zeros, then reveal initial pixels
        self.scan_mask.zero_()

        if self.rng_dis == 0:
            self.scan_mask[0, 0, int(self.x), int(self.y)] = 1.0
        else:
            self.scan_mask[
                0, 0,
                max(0, int(self.x) - self.rng_dis): min(int(self.x) + self.rng_dis + 1, self.len_x + 1),
                max(0, int(self.y) - self.rng_dis): min(int(self.y) + self.rng_dis + 1, self.len_y + 1),
            ] = 1.0

        self.step_count = 0
        self.predictions_history = []

        # masked_image is the agent's "current knowledge"
        self.masked_image = self.hidden_image * self.scan_mask + (self.scan_mask - 1).abs() * self.empty_image

        # x,y normalized in [0,1]
        self.current_state[0] = float(self.x) / float(self.len_x)
        self.current_state[1] = float(self.y) / float(self.len_y)

        self._write_state_patch()
        self._write_state_logits()

        self.step_size = 1
        self.counter = [0, 0, 0, 0]

        # clone() is important when you store states in replay buffer
        return self.current_state.clone()

    def step(self, action: int):
        """
        Execute one environment transition.

        The reward encourages:
          - discovering new pixels (explore_rwd * delta(masked_image))
          - avoiding bounces (bounce_rwd)
          - avoiding useless moves (return_rwd)
          - guessing correctly (hit_rwd) / incorrectly (miss_rwd)
          - taking fewer steps (step_rwd penalty each step)

        Returns:
          state (clone), reward (np.float32), done (bool), info
        """
        x0, y0 = int(self.x), int(self.y)
        self.step_count += 1

        info = "no info"

        # Snapshots are needed to detect "returned" and to compute "newly revealed info"
        img_before_step = self.masked_image.clone()
        mask_before_step = self.scan_mask.clone()

        # Jump step depends on how large an area is revealed each move
        jump = 1 + 2 * self.rng_dis
        x1, y1 = x0, y0

        # Movement actions update x1,y1; action==4 ("guess") leaves position unchanged
        if action == 0:  # up
            x1 = max(0, x0 - jump)
        elif action == 1:  # down
            x1 = min(self.size_x - 1, x0 + jump * self.step_size)
        elif action == 2:  # left
            y1 = max(0, y0 - jump * self.step_size)
        elif action == 3:  # right
            y1 = min(self.size_y - 1, y0 + jump * self.step_size)

        bounced = (x1 == x0 and y1 == y0)

        self.x = int(x1)
        self.y = int(y1)

        # Reveal new pixels around the new position
        if self.rng_dis == 0:
            self.scan_mask[0, 0, self.x, self.y] = 1.0
        else:
            self.scan_mask[
                0, 0,
                max(0, self.x - self.rng_dis): min(self.x + self.rng_dis + 1, self.len_x + 1),
                max(0, self.y - self.rng_dis): min(self.y + self.rng_dis + 1, self.len_y + 1),
            ] = 1.0

        # Update visible image after revealing
        self.masked_image = self.hidden_image * self.scan_mask + (self.scan_mask - 1).abs() * self.empty_image

        # Update state x,y
        self.current_state[0] = float(self.x) / float(self.len_x)
        self.current_state[1] = float(self.y) / float(self.len_y)

        self._write_state_patch()

        # returned=True means: scan_mask didn't change (no new pixels revealed), but you moved
        returned = torch.equal(mask_before_step, self.scan_mask) and (not bounced)

        # --------------------------------------------------------
        # Reward for navigation / exploration
        # --------------------------------------------------------
        if bounced:
            self.counter[0] += 1
            self.counter[2] += 1
            rwd = self.rwd_bounce
        elif returned:
            self.counter[1] += 1
            self.counter[3] += 1
            rwd = self.rwd_return
        else:
            # Reward proportional to newly revealed info (delta in masked image)
            rwd = (self.masked_image - img_before_step).sum().item() * self.rwd_explore

        done0 = (self.step_count >= self.nsteps)

        # Update logits part of state after the new scan is applied
        self._write_state_logits()

        # Explicit guess action terminates episode early
        if action == 4:
            done0 = True

        # --------------------------------------------------------
        # Terminal: compute final guess reward
        # --------------------------------------------------------
        if done0:
            with torch.no_grad():
                # Baseline: perceptron performance with the full image
                full_choice = int(torch.argmax(self.perceptron(self.hidden_image)[0]).item())
                guess_full = int(full_choice == int(self.hidden_label))

                # Agent guess uses only scanned pixels
                logits_now = self.perceptron(self.hidden_image * self.scan_mask)[0]
                choice = int(torch.argmax(logits_now).item())

            if choice == int(self.hidden_label):
                # Reward can depend on remaining steps (earlier correct guess -> higher reward)
                rwd = min(10, self.nsteps - self.step_count) * self.rwd_hit
                info = [1, guess_full]
                if self.verbose:
                    print(
                        f"{choice}, HIT nsteps:{self.step_count} "
                        f"bounces:{self.counter[0]} returns:{self.counter[1]} "
                        f"all_bounces:{self.counter[2]} all_returns:{self.counter[3]}"
                    )
            else:
                # Penalty scales with remaining steps (wrong guess earlier can be punished more)
                rwd = (self.nsteps - self.step_count) * self.rwd_miss
                info = [0, guess_full]
                if self.verbose:
                    print(
                        f"{choice}, -x- nsteps:{self.step_count} "
                        f"bounces:{self.counter[0]} returns:{self.counter[1]} "
                        f"all_bounces:{self.counter[2]} all_returns:{self.counter[3]}"
                    )

        # Per-step penalty encourages shorter trajectories regardless of outcome
        rwd += self.step_rwd

        return self.current_state.clone(), np.float32(rwd), bool(done0), info
