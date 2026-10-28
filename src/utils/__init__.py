"""Generic helpers, one module per role:

    distributed     process group, rank helpers, gathers, gradient averaging (sync_grads), LoRA broadcast
    rng             seeding, RNG state capture / restore
    batching        the training stage's sample order, timestep-axis views and micro-batches
    reward_gather   the epoch's rewards gathered over the ranks, zero-std ratio
"""
