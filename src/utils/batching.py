"""The epoch's samples as the training stage walks them: a shuffled order, a view of the timestep axis
(each sample's steps reordered, optionally only a band of them: a Wan2.2 expert's), and micro-batches."""
import torch


def permute(samples, perm):
    """Every per-sample tensor in ``perm`` order (other entries as they are)."""
    return {k: (v[perm] if isinstance(v, torch.Tensor) else v) for k, v in samples.items()}


def time_view(samples, perms_time, time_keys, band=None):
    """``samples`` with each ``time_keys`` entry reordered along the timestep axis, row i by
    ``perms_time[i]``; with ``band`` (bool mask over the steps) only those steps are kept first."""
    out = dict(samples)
    rows = torch.arange(perms_time.shape[0], device=perms_time.device)[:, None]
    for key in time_keys:
        if isinstance(out.get(key), torch.Tensor):
            x = out[key] if band is None or bool(band.all()) else out[key][:, band]
            out[key] = x[rows, perms_time]
    return out


def micro_batch(samples, b, size, device=None):
    """The b-th micro-batch of ``size`` samples; with ``device`` its tensors moved there (samples kept on
    CPU, config.cpu_offload_samples)."""
    start = b * size
    batch = {k: (v[start:start + size] if isinstance(v, torch.Tensor) else v) for k, v in samples.items()}
    if device is not None:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    return batch
