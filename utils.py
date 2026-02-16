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