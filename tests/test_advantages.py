"""src/advantage: the per-prompt advantage recipe (CPU, no model)."""
import unittest

import numpy as np
from ml_collections import ConfigDict

from src.advantage.advantages import PromptHistory, SignalSpec
from src.advantage.utils import group_signal, prompt_key, rank_rows, whitened_reward


def make_config(adv="standard", global_std=False, scale=1.0, beta=0.0, alpha=1.0, clip=5.0):
    return ConfigDict({"sample": {"global_std": global_std},
                       "train": {"adv": adv, "scale": scale, "beta": beta, "alpha": alpha, "adv_clip_max": clip}})


GROUPS = np.array(["a", "a", "a", "b", "b", "b"])
VALUES = np.array([1.0, 2.0, 3.0, 10.0, 10.0, 13.0])


class SignalSpecTests(unittest.TestCase):
    def test_rows(self):
        self.assertEqual(SignalSpec.from_adv(make_config()), SignalSpec(scale=1.0, sigma="prompt_std"))
        self.assertEqual(SignalSpec.from_adv(make_config(global_std=True)).sigma, "global_std")
        self.assertEqual(SignalSpec.from_adv(make_config("exact", beta=0.1, alpha=1e-3)),
                         SignalSpec(scale=0.1, sigma="const", sigma_const=1e-3))
        self.assertEqual(SignalSpec.from_adv(make_config("exact", scale=2.0)), SignalSpec(scale=2.0, sigma="const"))
        self.assertEqual(SignalSpec.from_adv(make_config("diffusionnft")).nft_clip, 5.0)
        with self.assertRaises(NotImplementedError):
            SignalSpec.from_adv(make_config("unknown"))


class GroupSignalTests(unittest.TestCase):
    def test_prompt_std(self):
        a = group_signal(GROUPS, VALUES, SignalSpec(scale=1.0, sigma="prompt_std"))
        for g in "ab":
            rows = VALUES[GROUPS == g]
            want = (rows - rows.mean()) / (rows.std() + 1e-4)
            np.testing.assert_allclose(a[GROUPS == g], want)
            self.assertAlmostEqual(a[GROUPS == g].sum(), 0.0)

    def test_global_std(self):
        a = group_signal(GROUPS, VALUES, SignalSpec(scale=2.0, sigma="global_std"))
        sigma = VALUES.std() + 1e-4
        for g in "ab":
            rows = VALUES[GROUPS == g]
            np.testing.assert_allclose(a[GROUPS == g], 2.0 * (rows - rows.mean()) / sigma)

    def test_const(self):
        a = group_signal(GROUPS, VALUES, SignalSpec(scale=0.1, sigma="const", sigma_const=1e-3))
        np.testing.assert_allclose(a[:3], 0.1 * np.array([-1.0, 0.0, 1.0]) / 1e-3)

    def test_nft_map_in_unit_interval(self):
        a = group_signal(GROUPS, VALUES, SignalSpec(scale=1.0, sigma="prompt_std", nft_clip=5.0))
        self.assertTrue(((a >= 0.0) & (a <= 1.0)).all())
        np.testing.assert_allclose(a[1], 0.5)          # the group mean maps to 0.5

    def test_per_column(self):
        cols = np.stack([VALUES, 2 * VALUES], axis=1)
        a = group_signal(GROUPS, cols, SignalSpec(scale=1.0, sigma="prompt_std"))
        self.assertEqual(a.shape, cols.shape)
        np.testing.assert_allclose(a[:, 0], a[:, 1], rtol=1e-4)


class HelperTests(unittest.TestCase):
    def test_rank_rows(self):
        values = np.arange(12).reshape(6, 2)
        np.testing.assert_array_equal(rank_rows(values, 2, 3, 1), values[3:])

    def test_whitened_reward(self):
        gathered = {"x": np.array([1.0, 2.0, 3.0]), "y": np.array([10.0, 30.0, 20.0])}
        w = whitened_reward(gathered, {"x": 1.0, "y": 0.5})
        zx = (gathered["x"] - 2.0) / (gathered["x"].std() + 1e-8)
        zy = (gathered["y"] - 20.0) / (gathered["y"].std() + 1e-8)
        np.testing.assert_allclose(w, zx + 0.5 * zy)

    def test_prompt_history(self):
        h = PromptHistory()
        h.add(["p", "q", "p"])
        self.assertEqual(h.history_prompts, {prompt_key("p"), prompt_key("q")})
        self.assertEqual(len(prompt_key("p")), 16)


if __name__ == "__main__":
    unittest.main()
