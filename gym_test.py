import sheepshead_v1
from gym_utils import train_action_mask, eval_action_mask

env_fn = sheepshead_v1

env_kwargs = {}

train_action_mask(env_fn, steps=10_240, seed=0, **env_kwargs)
