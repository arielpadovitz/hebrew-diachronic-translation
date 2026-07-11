"""
Stage 1b: extract contextual word embeddings using AlephBERT.

AlephBERT tokenizes into subwords, so each word's embedding is the mean of
its subword token embeddings, computed with the word in its full sentence
context (not in isolation).
"""

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from . import config


def load_alephbert():
    tokenizer = AutoTokenizer.from_pretrained(config.ALEPHBERT_MODEL)
    model = AutoModel.from_pretrained(config.ALEPHBERT_MODEL)
    model.eval()
    return tokenizer, model


def get_word_embeddings_from_sentence(sentence, num_words, tokenizer, model):
    """Return one embedding per word in `sentence` (a whitespace-joined string
    of `num_words` words), by averaging each word's subword token embeddings.

    Note: the original translate_and_analyze.ipynb had a copy of this function
    with `return word_embeddings` accidentally indented inside the loop, which
    silently truncated every call to just the first word's embedding. Fixed here.
    """
    inputs = tokenizer(sentence, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state.squeeze(0)

    word_ids = inputs.word_ids()
    word_embeddings = []

    for word_idx in range(num_words):
        token_indices = [i for i, wid in enumerate(word_ids) if wid == word_idx]

        if token_indices:
            word_emb = token_embeddings[token_indices].mean(dim=0).numpy()
        else:
            word_emb = np.zeros(model.config.hidden_size)

        word_embeddings.append(word_emb)

    return word_embeddings


def embed_corpus(df, tokenizer, model, log_every=100):
    """Add an 'embedding' column to `df`, one AlephBERT embedding per word,
    computed with each word's full sentence as context.
    """
    joined_sentences = df.groupby("sentence_num")["form"].apply(lambda words: " ".join(words))

    all_embeddings = []
    for i, sent_num in enumerate(joined_sentences.index):
        sentence = joined_sentences[sent_num]
        num_words = len(df[df["sentence_num"] == sent_num])

        embeddings = get_word_embeddings_from_sentence(sentence, num_words, tokenizer, model)
        all_embeddings.extend(embeddings)

        if (i + 1) % log_every == 0:
            print(f"  embedded {i + 1}/{len(joined_sentences)} sentences")

    df = df.copy()
    df["embedding"] = all_embeddings
    return df


def compute_embeddings(ancient_df, modern_df):
    """Run AlephBERT embedding extraction over both corpora."""
    tokenizer, model = load_alephbert()

    print("Embedding ancient corpus...")
    ancient_embedded = embed_corpus(ancient_df, tokenizer, model)

    print("Embedding modern corpus...")
    modern_embedded = embed_corpus(modern_df, tokenizer, model)

    return ancient_embedded, modern_embedded
