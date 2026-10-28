"""The epoch's rewards gathered over the ranks, and the per-prompt spread statistic."""
import numpy as np

from src.utils.distributed import gather_tensor_to_all

TIMESTEP_REWARDS = ("avg", "avg_low_noise", "avg_high_noise")   # repeated over the trained timesteps


def gather_epoch_rewards(samples, *, world_size, device, num_train_timesteps):
    """{key: [N] or [N, T] numpy} of every rank's rewards, rank-major; the combined rewards
    (TIMESTEP_REWARDS) repeated over the T trained timesteps, as the advantages are computed per
    timestep. The gather is NCCL: it runs on a ``device`` copy."""
    out = {}
    for key, value in samples["rewards"].items():
        value = value.to(device)
        if key in TIMESTEP_REWARDS:
            value = value.unsqueeze(1).repeat(1, num_train_timesteps)
        out[key] = gather_tensor_to_all(value, world_size).numpy()
    return out


def calculate_zero_std_ratio(prompts, gathered_rewards):
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(prompt_array, return_inverse=True, return_counts=True)
    grouped_rewards = gathered_rewards["avg"][np.argsort(inverse_indices), 0]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    return zero_std_ratio, prompt_std_devs.mean()
