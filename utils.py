import random
import numpy as np
import torch

def set_global_seed(seed: int = 42):
    """
    Make runs reproducible by fixing RNG seeds in Python, NumPy and Torch.

    Note: Setting cudnn.deterministic=True can slow GPU training a bit,
    but gives repeatable results.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_state(observation) -> torch.Tensor:
    """
    Convert Env observation into a 1D float32 torch.Tensor on CPU.

    Why: your memory buffer stores states, so you want a consistent dtype/shape
    regardless of whether Env returns torch / numpy / list.
    Returns shape: (obs_dim,)
    """
    if isinstance(observation, torch.Tensor):
        return observation.detach().clone().to(dtype=torch.float32).view(-1)
    if isinstance(observation, np.ndarray):
        return torch.from_numpy(observation).to(dtype=torch.float32).view(-1)
    return torch.tensor(observation, dtype=torch.float32).view(-1)

import torch

import torch
from collections import OrderedDict

import torch

import torch

def load_or_keep(model, fname: str):
    """
    Returns a state_dict you can always pass to model.load_state_dict(...).

    - If file doesn't exist -> returns model.state_dict() (no-op)
    - If checkpoint mismatches (keys or shapes) -> returns model.state_dict() (no-op)
    - If compatible -> returns checkpoint state_dict
    """
    current = model.state_dict()

    # 1) File missing
    try:
        sd = torch.load(fname, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        return current

    # 2) Mismatch (keys/shapes) detected by load_state_dict
    try:
        # strict=True is default; it will raise on size mismatch / missing keys / unexpected keys
        model.load_state_dict(sd, strict=True)
        return sd
    except RuntimeError:
        return current