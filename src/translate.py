"""
Stage 4: use the trained grammar embeddings to "translate" ancient words to
their nearest modern equivalent under three embedding configurations, then
evaluate translation quality by re-embedding the translated sentences and
comparing to the original ancient contextual embeddings.

Configurations:
  word:  AlephBERT embedding only
  pos:   AlephBERT + POS embedding
  full:  AlephBERT + POS + head-POS + dependency embedding
"""

from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from . import config
from .embed import get_word_embeddings_from_sentence, load_alephbert

CONFIGS = ("word", "pos", "full")


def build_config_matrix(df, trained_embeddings, use_pos=False, use_head_pos=False, use_dep=False):
    """Concatenate AlephBERT embeddings with the requested grammar embeddings,
    L2-normalized row-wise so cosine similarity reduces to a dot product.
    """
    alephbert = np.vstack(df["embedding"].values)
    parts = [alephbert]

    if use_pos:
        idx = df["upos"].map(trained_embeddings["pos_to_idx"]).values
        parts.append(trained_embeddings["pos_embeddings"][idx])
    if use_head_pos:
        idx = df["head_pos"].map(trained_embeddings["head_pos_to_idx"]).values
        parts.append(trained_embeddings["head_pos_embeddings"][idx])
    if use_dep:
        idx = df["deprel"].map(trained_embeddings["dep_to_idx"]).values
        parts.append(trained_embeddings["dep_embeddings"][idx])

    matrix = np.concatenate(parts, axis=1)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def build_all_config_matrices(ancient_df, modern_df, trained_embeddings):
    return {
        "word": (
            build_config_matrix(ancient_df, trained_embeddings),
            build_config_matrix(modern_df, trained_embeddings),
        ),
        "pos": (
            build_config_matrix(ancient_df, trained_embeddings, use_pos=True),
            build_config_matrix(modern_df, trained_embeddings, use_pos=True),
        ),
        "full": (
            build_config_matrix(ancient_df, trained_embeddings, use_pos=True, use_head_pos=True, use_dep=True),
            build_config_matrix(modern_df, trained_embeddings, use_pos=True, use_head_pos=True, use_dep=True),
        ),
    }


def sample_ancient_sentences(ancient_df, fraction=config.EVAL_SAMPLE_FRACTION, seed=config.RANDOM_SEED):
    rng = np.random.default_rng(seed)
    all_sentences = ancient_df["sentence_num"].unique()
    n = int(len(all_sentences) * fraction)
    return rng.choice(all_sentences, n, replace=False)


def translate_sentences(ancient_df, modern_df, config_matrices, sample_sentence_nums):
    """For each sampled ancient sentence, translate every word under each
    embedding configuration by nearest-neighbor cosine similarity in modern space.
    """
    translations = {}

    for sent_num in sample_sentence_nums:
        sent_rows = ancient_df[ancient_df["sentence_num"] == sent_num].sort_values("id")
        if len(sent_rows) == 0:
            continue

        matrix_positions = [ancient_df.index.get_loc(idx) for idx in sent_rows.index]
        translations[sent_num] = {
            "ancient_words": sent_rows["form"].tolist(),
            "matrix_positions": matrix_positions,
        }

        for config_name in CONFIGS:
            ancient_matrix, modern_matrix = config_matrices[config_name]
            anchor_embs = ancient_matrix[matrix_positions]
            similarities = anchor_embs @ modern_matrix.T
            best_indices = similarities.argmax(axis=1)
            translations[sent_num][config_name] = modern_df.iloc[best_indices]["form"].tolist()

    return translations


def evaluate_translations(ancient_df, translations, tokenizer, model):
    """Re-embed each translated sentence with AlephBERT and compare, word by
    word, to the original ancient contextual embedding. Returns both overall
    and POS-conditioned average cosine similarity per configuration.
    """
    overall = {c: [] for c in CONFIGS}
    by_pos = defaultdict(lambda: defaultdict(list))

    for sent_num, sent_data in translations.items():
        matrix_positions = sent_data["matrix_positions"]
        ancient_embs = ancient_df.iloc[matrix_positions]["embedding"].tolist()
        ancient_pos_tags = ancient_df.iloc[matrix_positions]["upos"].tolist()

        for config_name in CONFIGS:
            modern_words = sent_data[config_name]
            modern_sentence = " ".join(modern_words)
            modern_embs = get_word_embeddings_from_sentence(
                modern_sentence, len(modern_words), tokenizer, model
            )

            for anc_emb, mod_emb, pos_tag in zip(ancient_embs, modern_embs, ancient_pos_tags):
                sim = cosine_similarity([anc_emb], [mod_emb])[0][0]
                overall[config_name].append(sim)
                by_pos[pos_tag][config_name].append(sim)

    overall_avg = {c: float(np.mean(v)) for c, v in overall.items()}

    pos_avg = {}
    for pos_tag, config_scores in by_pos.items():
        pos_avg[pos_tag] = {
            "count": len(config_scores["word"]),
            **{c: float(np.mean(v)) for c, v in config_scores.items()},
        }

    return {"overall": overall_avg, "by_pos": pos_avg}


def run_evaluation(ancient_df, modern_df, trained_embeddings):
    """Full pipeline: build config matrices, sample sentences, translate, evaluate."""
    config_matrices = build_all_config_matrices(ancient_df, modern_df, trained_embeddings)
    sample_sentence_nums = sample_ancient_sentences(ancient_df)
    translations = translate_sentences(ancient_df, modern_df, config_matrices, sample_sentence_nums)

    tokenizer, model = load_alephbert()
    results = evaluate_translations(ancient_df, translations, tokenizer, model)

    return translations, results
