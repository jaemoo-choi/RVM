from .rewards import multi_score

__all__ = ["multi_score"]


def get_reward_fn(device, reward_config, videoalign_dimension_weights=None, video_fps=None):
    return multi_score(
        device, reward_config,
        videoalign_dimension_weights=videoalign_dimension_weights,
        video_fps=video_fps,
    )
