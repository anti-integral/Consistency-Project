import random, os, torch, numpy as np
def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)

def save_checkpoint(state_dict, path):
    torch.save(state_dict, path)
