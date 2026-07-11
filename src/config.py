"""
Central configuration for the pipeline: file paths, URLs, and hyperparameters.

This is the fix for the original notebooks' biggest pain point: every notebook
independently mounted Google Drive and hardcoded its own copy of the same paths,
so a path change meant editing four files. Now every stage imports these paths
from one place.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
UD_DATA_DIR = os.path.join(DATA_DIR, "ud_data")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")

for _dir in (DATA_DIR, UD_DATA_DIR, EMBEDDINGS_DIR):
    os.makedirs(_dir, exist_ok=True)

# --- Model ---
ALEPHBERT_MODEL = "onlplab/alephbert-base"

# --- Source data (Universal Dependencies treebanks) ---
UD_URLS = {
    "ancient_train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Ancient_Hebrew-PTNK/master/hbo_ptnk-ud-train.conllu",
    "ancient_dev": "https://raw.githubusercontent.com/UniversalDependencies/UD_Ancient_Hebrew-PTNK/master/hbo_ptnk-ud-dev.conllu",
    "ancient_test": "https://raw.githubusercontent.com/UniversalDependencies/UD_Ancient_Hebrew-PTNK/master/hbo_ptnk-ud-test.conllu",
    "modern_train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Hebrew-HTB/refs/heads/master/he_htb-ud-train.conllu",
    "modern_dev": "https://raw.githubusercontent.com/UniversalDependencies/UD_Hebrew-HTB/refs/heads/master/he_htb-ud-dev.conllu",
    "modern_test": "https://raw.githubusercontent.com/UniversalDependencies/UD_Hebrew-HTB/refs/heads/master/he_htb-ud-test.conllu",
}

# --- Cached intermediate artifacts (this replaces scattered Drive re-reads) ---
ANCIENT_EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "ancient_embeddings.pkl")
MODERN_EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "modern_embeddings.pkl")
COMBINED_EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "combined_embeddings.pkl")
TRIPLETS_PATH = os.path.join(EMBEDDINGS_DIR, "triplets.pkl")
TRAINED_GRAMMAR_EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "trained_grammar_embeddings.pkl")
TRANSLATIONS_PATH = os.path.join(EMBEDDINGS_DIR, "translations.pkl")
RESULTS_PATH = os.path.join(EMBEDDINGS_DIR, "evaluation_results.pkl")

# --- Reproducibility ---
RANDOM_SEED = 42

# --- Contrastive training hyperparameters ---
TRIPLET_MARGIN = 1.0
LEARNING_RATE = 0.001
EPOCHS = 60
BATCH_SIZE = 512

# Grammatical embedding dimensions (chosen arbitrarily in the original project;
# noted in the report as a candidate for future tuning)
POS_EMBED_DIM = 128
HEAD_POS_EMBED_DIM = 64
DEP_EMBED_DIM = 64

# Fixed vocabulary sizes for the grammar embedding tables, derived from the
# Universal Dependencies tagset used across both corpora.
NUM_POS_TAGS = 16
NUM_HEAD_POS_TAGS = 17
NUM_DEP_RELS = 42

# Fraction of the ancient corpus sampled for translation evaluation
EVAL_SAMPLE_FRACTION = 0.2
