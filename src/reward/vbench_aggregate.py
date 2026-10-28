"""Official VBench / VBench++ leaderboard aggregation.

The pip `vbench==0.1.5` package does NOT ship `scripts/constant.py` or
`scripts/cal_final_score.py`, so the official leaderboard math (per-dimension
min-max normalization, the `dynamic_degree` 0.5 weight, the 4:1 Quality:Semantic
combination) is not importable. We vendor the exact constants and reproduce the
aggregation here so our reported Total / Quality / Semantic match the public
VBench leaderboard.

Constants are copied verbatim from
github.com/Vchitect/VBench `scripts/constant.py` (T2V) and the I2V equivalents
used by `scripts/cal_i2v_final_score.py`.

Key facts this module encodes (these are the things our old 0.55/0.45 plain-mean
aggregation got wrong):
  * Each dimension is min-max normalized with fixed (Min, Max) BEFORE averaging.
  * `dynamic_degree` is weighted 0.5 (everything else 1.0) in T2V; in I2V
    `camera_motion` is weighted 0.1 and `dynamic_degree` 0.5.
  * T2V Total = (4 * Quality + 1 * Semantic) / 5  ==  0.8 * Q + 0.2 * S.
  * Quality is the mean over 7 dims (denominator 6.5 because of dynamic_degree),
    Semantic is the mean over 9 dims (denominator 9.0).
  * I2V Total = (Quality + I2V) / 2 (equal 1.0 / 1.0 weights). `temporal_flickering`
    is computed by VBench-I2V but is NOT part of either group (absent from
    TASK_INFO_I2V), so it never enters the I2V Total.

All functions accept the *underscore* dimension names that
`vbench.VBench.evaluate()` / `VBenchI2V.evaluate()` produce (e.g.
"subject_consistency", "dynamic_degree", "i2v_subject") and return their results
keyed the same way, so callers never deal with the leaderboard's spaced display
names.
"""

from __future__ import annotations

# ----------------------------------------------------------------------------
# T2V constants (verbatim from VBench scripts/constant.py)
# ----------------------------------------------------------------------------
# Keyed by the leaderboard's spaced display names. We translate from the
# underscore dim names with a simple "_" -> " " replacement, which is exact for
# all 16 T2V dimensions.
NORMALIZE_DIC = {
    "subject consistency": {"Min": 0.1462, "Max": 1.0},
    "background consistency": {"Min": 0.2615, "Max": 1.0},
    "temporal flickering": {"Min": 0.6293, "Max": 1.0},
    "motion smoothness": {"Min": 0.706, "Max": 0.9975},
    "dynamic degree": {"Min": 0.0, "Max": 1.0},
    "aesthetic quality": {"Min": 0.0, "Max": 1.0},
    "imaging quality": {"Min": 0.0, "Max": 1.0},
    "object class": {"Min": 0.0, "Max": 1.0},
    "multiple objects": {"Min": 0.0, "Max": 1.0},
    "human action": {"Min": 0.0, "Max": 1.0},
    "color": {"Min": 0.0, "Max": 1.0},
    "spatial relationship": {"Min": 0.0, "Max": 1.0},
    "scene": {"Min": 0.0, "Max": 0.8222},
    "appearance style": {"Min": 0.0009, "Max": 0.2855},
    "temporal style": {"Min": 0.0, "Max": 0.364},
    "overall consistency": {"Min": 0.0, "Max": 0.364},
}

DIM_WEIGHT = {
    "subject consistency": 1,
    "background consistency": 1,
    "temporal flickering": 1,
    "motion smoothness": 1,
    "aesthetic quality": 1,
    "imaging quality": 1,
    "dynamic degree": 0.5,
    "object class": 1,
    "multiple objects": 1,
    "human action": 1,
    "color": 1,
    "spatial relationship": 1,
    "scene": 1,
    "appearance style": 1,
    "temporal style": 1,
    "overall consistency": 1,
}

QUALITY_LIST = [
    "subject consistency",
    "background consistency",
    "temporal flickering",
    "motion smoothness",
    "aesthetic quality",
    "imaging quality",
    "dynamic degree",
]

SEMANTIC_LIST = [
    "object class",
    "multiple objects",
    "human action",
    "color",
    "spatial relationship",
    "scene",
    "appearance style",
    "temporal style",
    "overall consistency",
]

QUALITY_WEIGHT = 4
SEMANTIC_WEIGHT = 1

# ----------------------------------------------------------------------------
# I2V constants (verbatim from VBench scripts/constant.py)
# ----------------------------------------------------------------------------
NORMALIZE_DIC_I2V = {
    "Video-Text Camera Motion": {"Min": 0.0, "Max": 1.0},
    "Video-Image Subject Consistency": {"Min": 0.1462, "Max": 1.0},
    "Video-Image Background Consistency": {"Min": 0.2615, "Max": 1.0},
    "Subject Consistency": {"Min": 0.1462, "Max": 1.0},
    "Background Consistency": {"Min": 0.2615, "Max": 1.0},
    "Motion Smoothness": {"Min": 0.7060, "Max": 0.9975},
    "Dynamic Degree": {"Min": 0.0, "Max": 1.0},
    "Aesthetic Quality": {"Min": 0.0, "Max": 1.0},
    "Imaging Quality": {"Min": 0.0, "Max": 1.0},
}

DIM_WEIGHT_I2V = {
    "Video-Text Camera Motion": 0.1,
    "Video-Image Subject Consistency": 1,
    "Video-Image Background Consistency": 1,
    "Subject Consistency": 1,
    "Background Consistency": 1,
    "Motion Smoothness": 1,
    "Dynamic Degree": 0.5,
    "Aesthetic Quality": 1,
    "Imaging Quality": 1,
}

I2V_LIST = [
    "Video-Text Camera Motion",
    "Video-Image Subject Consistency",
    "Video-Image Background Consistency",
]

I2V_QUALITY_LIST = [
    "Subject Consistency",
    "Background Consistency",
    "Motion Smoothness",
    "Dynamic Degree",
    "Aesthetic Quality",
    "Imaging Quality",
]

I2V_WEIGHT = 1.0
I2V_QUALITY_WEIGHT = 1.0

# Maps the underscore dim names that VBenchI2V.evaluate() returns to the spaced
# display names the constant dicts above use. (For T2V a plain "_"->" " replace
# is exact; for I2V the i2v_* / *_consistency names need this explicit table.)
I2VKEY = {
    "camera_motion": "Video-Text Camera Motion",
    "i2v_subject": "Video-Image Subject Consistency",
    "i2v_background": "Video-Image Background Consistency",
    "subject_consistency": "Subject Consistency",
    "background_consistency": "Background Consistency",
    "motion_smoothness": "Motion Smoothness",
    "dynamic_degree": "Dynamic Degree",
    "aesthetic_quality": "Aesthetic Quality",
    "imaging_quality": "Imaging Quality",
}

# Underscore-named convenience sets for callers that group dims without going
# through the leaderboard aggregation (kept so existing imports keep working).
QUALITY_DIMS = {d.replace(" ", "_") for d in QUALITY_LIST}
SEMANTIC_DIMS = {d.replace(" ", "_") for d in SEMANTIC_LIST}


def _normalize(value: float, key: str, norm_dic: dict, weight_dic: dict) -> float:
    """Min-max normalize then apply the dimension weight (official formula)."""
    lo = norm_dic[key]["Min"]
    hi = norm_dic[key]["Max"]
    return ((value - lo) / (hi - lo)) * weight_dic[key]


def leaderboard_scores_t2v(raw_per_dim_avg: dict) -> dict:
    """Official VBench T2V leaderboard aggregation.

    Args:
        raw_per_dim_avg: {underscore_dim_name: raw_average_in_[0,1]}. Only the
            dims actually present are aggregated (so a partial run still yields
            a Quality/Semantic/Total over whatever scored), with each group's
            denominator restricted to the present dims' weights — a graceful
            degradation of the official full-suite formula.

    Returns:
        {
          "raw_per_dim": {dim: raw},
          "normalized_per_dim": {dim: normalized*weight},
          "Quality": float | None,
          "Semantic": float | None,
          "Total": float | None,
          "scored_dims": [...], "missing_dims": [...],
        }
    """
    normalized = {}
    raw = {}
    for dim, value in raw_per_dim_avg.items():
        disp = dim.replace("_", " ")
        if disp not in NORMALIZE_DIC:
            continue
        raw[dim] = float(value)
        normalized[dim] = _normalize(float(value), disp, NORMALIZE_DIC, DIM_WEIGHT)

    def _group(dim_list_disp):
        present = [d for d in dim_list_disp if d.replace(" ", "_") in normalized]
        if not present:
            return None
        num = sum(normalized[d.replace(" ", "_")] for d in present)
        den = sum(DIM_WEIGHT[d] for d in present)
        return num / den if den else None

    quality = _group(QUALITY_LIST)
    semantic = _group(SEMANTIC_LIST)

    if quality is not None and semantic is not None:
        total = (quality * QUALITY_WEIGHT + semantic * SEMANTIC_WEIGHT) / (
            QUALITY_WEIGHT + SEMANTIC_WEIGHT
        )
    elif quality is not None:
        total = quality
    elif semantic is not None:
        total = semantic
    else:
        total = None

    all_dims = {d.replace(" ", "_") for d in QUALITY_LIST + SEMANTIC_LIST}
    return {
        "raw_per_dim": raw,
        "normalized_per_dim": normalized,
        "Quality": quality,
        "Semantic": semantic,
        "Total": total,
        "scored_dims": sorted(raw.keys()),
        "missing_dims": sorted(all_dims - set(raw.keys())),
    }


def leaderboard_scores_i2v(raw_per_dim_avg: dict) -> dict:
    """Official VBench++ (VBench-I2V) leaderboard aggregation.

    Args:
        raw_per_dim_avg: {underscore_dim_name: raw_average}. `temporal_flickering`
            is ignored (not part of the I2V leaderboard groups). Partial runs are
            handled the same way as the T2V helper.

    Returns:
        {raw_per_dim, normalized_per_dim, "I2V": float|None,
         "Quality": float|None, "Total": float|None, scored_dims, missing_dims}
    """
    normalized = {}
    raw = {}
    for dim, value in raw_per_dim_avg.items():
        disp = I2VKEY.get(dim)
        if disp is None or disp not in NORMALIZE_DIC_I2V:
            continue  # e.g. temporal_flickering -> not aggregated
        raw[dim] = float(value)
        normalized[dim] = _normalize(
            float(value), disp, NORMALIZE_DIC_I2V, DIM_WEIGHT_I2V
        )

    inv_key = {v: k for k, v in I2VKEY.items()}

    def _group(dim_list_disp):
        present = [d for d in dim_list_disp if inv_key[d] in normalized]
        if not present:
            return None
        num = sum(normalized[inv_key[d]] for d in present)
        den = sum(DIM_WEIGHT_I2V[d] for d in present)
        return num / den if den else None

    i2v = _group(I2V_LIST)
    quality = _group(I2V_QUALITY_LIST)

    if i2v is not None and quality is not None:
        total = (quality * I2V_QUALITY_WEIGHT + i2v * I2V_WEIGHT) / (
            I2V_QUALITY_WEIGHT + I2V_WEIGHT
        )
    elif quality is not None:
        total = quality
    elif i2v is not None:
        total = i2v
    else:
        total = None

    all_dims = {inv_key[d] for d in I2V_LIST + I2V_QUALITY_LIST}
    return {
        "raw_per_dim": raw,
        "normalized_per_dim": normalized,
        "I2V": i2v,
        "Quality": quality,
        "Total": total,
        "scored_dims": sorted(raw.keys()),
        "missing_dims": sorted(all_dims - set(raw.keys())),
    }
