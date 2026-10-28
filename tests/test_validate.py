"""src/engine/checkpoint.py: the trainers' startup check of a config (CPU, no model)."""
import unittest

from ml_collections import ConfigDict

from config.validate import validate_train_config


def make_config(base_model="wan2_2_a14b_t2v", **train):
    c = ConfigDict()
    c.base_model = base_model
    c.train = ConfigDict(dict(method="rvm", **train))
    return c


class WhitenRewardsTests(unittest.TestCase):
    def test_wan2_2_rejects_whiten_rewards(self):
        with self.assertRaisesRegex(ValueError, r"train\.whiten_rewards=True: src/train_wan2_2\.py"):
            validate_train_config(make_config(whiten_rewards=True), "wan2_2")

    def test_wan2_2_accepts_whiten_off_or_absent(self):
        validate_train_config(make_config(), "wan2_2")
        validate_train_config(make_config(whiten_rewards=False), "wan2_2")

    def test_velocity_keeps_whiten_rewards(self):
        validate_train_config(make_config("skyreels_i2v", whiten_rewards=True), "velocity")


def accum_config(base_model="wan2_1", sbs=8, nbpe=4, bs=8, ga=2, group=None, **train):
    c = make_config(base_model, batch_size=bs, gradient_accumulation_steps=ga, **train)
    c.sample = ConfigDict(dict(train_batch_size=sbs, num_batches_per_epoch=nbpe))
    if group is not None:
        c.sample.num_image_per_prompt = group
    return c


class PartialAccumulationTests(unittest.TestCase):
    """num_batches % gradient_accumulation_steps != 0 would carry the epoch's leftover gradients
    into the next epoch's first optimizer step (src/train.py); no kept launcher has it, so it is
    rejected at startup."""

    def test_kept_shapes_pass(self):
        validate_train_config(accum_config(), "velocity")                                   # Wan2.1 4 / 2
        validate_train_config(accum_config("sd3", sbs=9, nbpe=16, bs=3, ga=24), "velocity")  # 48 / 24

    def test_leftover_rejected(self):
        with self.assertRaisesRegex(ValueError, r"gradient_accumulation_steps=3 does not divide the 4 micro-batches"):
            validate_train_config(accum_config(ga=3), "velocity")

    def test_wan2_2_and_incomplete_configs_skip(self):
        validate_train_config(accum_config("wan2_2_a14b_t2v", ga=3), "wan2_2")   # has its own runtime assert
        validate_train_config(make_config("wan2_1"), "velocity")                 # no sample fields: skipped


if __name__ == "__main__":
    unittest.main()
