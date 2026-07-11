"""
Stage 2: build contrastive triplets (anchor, positive, negative) across both
corpora, grouped by grammatical profile (POS tag + dependency relation + head's
POS tag).

Three triplet types are built, matching the project report:
  Type A: ancient anchor, modern positive (same grammar), modern negative (different grammar)
  Type B: ancient anchor, ancient positive (same grammar), modern negative (different grammar)
  Type C: ancient anchor, modern positive (same grammar), ancient negative (different grammar)

Note: the original contrastive_triplets.ipynb defined add_head_pos_vectorized
with the inner function and its return statement dedented to module level,
which only worked by accident of how the notebook cell executed. Rewritten
here as a correctly nested function.
"""

import random

import numpy as np
import pandas as pd

from . import config


def add_head_pos(df):
    """For each token, look up its syntactic head's POS tag within the same sentence."""
    lookup = df.set_index(["sentence_num", "id"])["upos"].to_dict()

    def head_pos_for_row(row):
        if row["head"] == "0":
            return "ROOT"
        key = (row["sentence_num"], row["head"])
        return lookup.get(key, "NONE")

    return df.apply(head_pos_for_row, axis=1)


def build_combined_df(ancient_df, modern_df):
    """Tag each corpus, attach head-POS, and concatenate into one DataFrame
    with a group_key column identifying each word's grammatical profile.
    """
    ancient_df = ancient_df.copy()
    modern_df = modern_df.copy()

    ancient_df["corpus"] = "ancient"
    modern_df["corpus"] = "modern"

    ancient_df["head_pos"] = add_head_pos(ancient_df)
    modern_df["head_pos"] = add_head_pos(modern_df)

    combined_df = pd.concat([ancient_df, modern_df], ignore_index=True)
    combined_df["group_key"] = (
        combined_df["upos"] + "_" + combined_df["deprel"] + "_" + combined_df["head_pos"]
    )
    return combined_df


def _group_indices_by_key(df):
    """Map each group_key to a numpy array of matching row indices."""
    groups = {}
    for idx, group_key in zip(df.index, df["group_key"]):
        groups.setdefault(group_key, []).append(idx)
    return {k: np.array(v) for k, v in groups.items()}


def build_triplets(combined_df, seed=config.RANDOM_SEED):
    """Construct type A/B/C contrastive triplets. Returns a shuffled list of
    (anchor_idx, positive_idx, negative_idx) tuples, indexing into combined_df.
    """
    random.seed(seed)
    np.random.seed(seed)

    ancient_df = combined_df[combined_df["corpus"] == "ancient"]
    modern_df = combined_df[combined_df["corpus"] == "modern"]

    ancient_groups = _group_indices_by_key(ancient_df)
    modern_groups = _group_indices_by_key(modern_df)

    shared_groups = set(ancient_groups) & set(modern_groups)
    all_modern_indices = modern_df.index.to_numpy()

    print(f"Ancient unique groups: {len(ancient_groups)}")
    print(f"Modern unique groups:  {len(modern_groups)}")
    print(f"Shared groups:         {len(shared_groups)}")

    type_a, type_b, type_c = [], [], []

    # Type A: ancient anchor, modern positive, modern negative
    for group_key in ancient_groups:
        if group_key not in shared_groups:
            continue
        group_indices = ancient_groups[group_key]
        modern_same = modern_groups.get(group_key, np.array([]))
        if len(modern_same) == 0:
            continue
        modern_diff = np.setdiff1d(all_modern_indices, modern_same)
        if len(modern_diff) == 0:
            continue

        for anchor_idx in group_indices:
            pos_idx = np.random.choice(modern_same)
            neg_idx = np.random.choice(modern_diff)
            type_a.append((anchor_idx, pos_idx, neg_idx))

    # Type B: ancient anchor, ancient positive, modern negative
    for group_key, group_indices in ancient_groups.items():
        if len(group_indices) < 2:
            continue
        modern_diff_groups = modern_df[modern_df["group_key"] != group_key].index.to_numpy()
        if len(modern_diff_groups) == 0:
            continue

        for i, anchor_idx in enumerate(group_indices):
            other_ancient = np.delete(group_indices, i)
            pos_idx = np.random.choice(other_ancient)
            neg_idx = np.random.choice(modern_diff_groups)
            type_b.append((anchor_idx, pos_idx, neg_idx))

    # Type C: ancient anchor, modern positive, ancient negative
    for group_key in ancient_groups:
        if group_key not in shared_groups:
            continue
        group_indices = ancient_groups[group_key]
        modern_same = modern_groups.get(group_key, np.array([]))
        if len(modern_same) == 0:
            continue
        ancient_diff_groups = ancient_df[ancient_df["group_key"] != group_key].index.to_numpy()
        if len(ancient_diff_groups) == 0:
            continue

        for anchor_idx in group_indices:
            pos_idx = np.random.choice(modern_same)
            neg_candidates = ancient_diff_groups[ancient_diff_groups != anchor_idx]
            if len(neg_candidates) == 0:
                continue
            neg_idx = np.random.choice(neg_candidates)
            type_c.append((anchor_idx, pos_idx, neg_idx))

    print(f"{len(type_a)} type-A triplets")
    print(f"{len(type_b)} type-B triplets")
    print(f"{len(type_c)} type-C triplets")

    all_triplets = type_a + type_b + type_c
    random.shuffle(all_triplets)
    return all_triplets
