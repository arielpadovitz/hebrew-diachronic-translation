"""
Run the full ancient-to-modern Hebrew translation pipeline, or a single stage
of it, from the command line.

    python run_pipeline.py                  # run everything, using cached results where available
    python run_pipeline.py --stage embed     # run only through embedding extraction
    python run_pipeline.py --force           # ignore all caches and recompute everything

Each stage's output is cached to disk (see src/utils.py:load_or_compute), so
re-running this script after, say, changing the training hyperparameters will
skip straight to re-training rather than re-downloading data and re-running
AlephBERT over both corpora again.
"""

import argparse
import os
import pickle
import pprint

from src import config, data, embed, train, translate, triplets
from src.utils import load_or_compute

STAGES = ["data", "embed", "triplets", "train", "evaluate"]


def run(stage="evaluate", force=False):
    stage_index = STAGES.index(stage)

    # Stage 1a: raw corpora
    ancient_df, modern_df = data.load_corpora()
    if stage_index == STAGES.index("data"):
        return ancient_df, modern_df

    # Stage 1b: AlephBERT embeddings
    def compute_embeddings():
        return embed.compute_embeddings(ancient_df, modern_df)

    ancient_embedded_path = config.ANCIENT_EMBEDDINGS_PATH
    modern_embedded_path = config.MODERN_EMBEDDINGS_PATH

    # embeddings are cached as a pair, so cache/recompute them together
    if not force and os.path.exists(ancient_embedded_path) and os.path.exists(modern_embedded_path):
        print("[cache hit] loading embedded corpora")
        with open(ancient_embedded_path, "rb") as f:
            ancient_embedded = pickle.load(f)
        with open(modern_embedded_path, "rb") as f:
            modern_embedded = pickle.load(f)
    else:
        ancient_embedded, modern_embedded = compute_embeddings()
        with open(ancient_embedded_path, "wb") as f:
            pickle.dump(ancient_embedded, f)
        with open(modern_embedded_path, "wb") as f:
            pickle.dump(modern_embedded, f)

    if stage_index == STAGES.index("embed"):
        return ancient_embedded, modern_embedded

    # Stage 2: contrastive triplets
    combined_df = triplets.build_combined_df(ancient_embedded, modern_embedded)
    all_triplets = load_or_compute(
        config.TRIPLETS_PATH,
        lambda: triplets.build_triplets(combined_df),
        force_recompute=force,
    )
    if stage_index == STAGES.index("triplets"):
        return combined_df, all_triplets

    # Stage 3: train grammar embeddings
    trained_embeddings = load_or_compute(
        config.TRAINED_GRAMMAR_EMBEDDINGS_PATH,
        lambda: train.train_grammar_embeddings(combined_df, all_triplets),
        force_recompute=force,
    )
    if stage_index == STAGES.index("train"):
        return trained_embeddings

    # Stage 4: translate + evaluate
    ancient_only = combined_df[combined_df["corpus"] == "ancient"].reset_index(drop=True)
    modern_only = combined_df[combined_df["corpus"] == "modern"].reset_index(drop=True)

    translations, results = translate.run_evaluation(ancient_only, modern_only, trained_embeddings)

    print("\n=== Overall average cosine similarity by configuration ===")
    pprint.pprint(results["overall"])

    return translations, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=STAGES, default="evaluate", help="run up to and including this stage")
    parser.add_argument("--force", action="store_true", help="ignore cached results and recompute everything")
    args = parser.parse_args()

    run(stage=args.stage, force=args.force)
