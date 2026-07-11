"""
Stage 1a: acquire and parse the raw Universal Dependencies data.

Downloads the ancient Hebrew (UD_Ancient_Hebrew-PTNK) and modern Hebrew
(UD_Hebrew-HTB) treebanks, parses the CoNLL-U format into plain Python
structures, and builds one DataFrame per corpus.
"""

import os
import re
import unicodedata

import pandas as pd
import requests

from . import config


def download_ud_data(force_redownload=False):
    """Download all CoNLL-U files listed in config.UD_URLS, if not already present."""
    paths = {}
    for name, url in config.UD_URLS.items():
        file_path = os.path.join(config.UD_DATA_DIR, f"{name}.conllu")
        paths[name] = file_path

        if os.path.exists(file_path) and not force_redownload:
            continue

        print(f"Downloading {name}...")
        response = requests.get(url)
        response.raise_for_status()
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(response.text)

    return paths


def parse_conllu(file_path):
    """Parse a CoNLL-U file into a list of sentences, each a list of token dicts."""
    sentences = []
    sentence = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line == "":
                if sentence:
                    sentences.append(sentence)
                    sentence = []
                continue

            if line.startswith("#"):
                continue

            cols = line.split("\t")

            # skip multiword-token range lines (e.g. "3-4")
            if "-" in cols[0]:
                continue
            if len(cols) < 10:
                continue

            token = {
                "id": cols[0],
                "form": cols[1],
                "lemma": cols[2],
                "upos": cols[3],
                "xpos": cols[4],
                "feats": cols[5],
                "head": cols[6],
                "deprel": cols[7],
                "misc": cols[9],
            }
            sentence.append(token)

    if sentence:
        sentences.append(sentence)

    return sentences


def remove_hebrew_diacritics(text):
    """Strip Hebrew vowel points (niqqud) and cantillation marks, normalize the result."""
    text = re.sub(r"[\u0591-\u05AF]", "", text)  # cantillation
    text = re.sub(r"[\u05B0-\u05BC\u05C1\u05C2]", "", text)  # vowel points
    text = re.sub(r"[\u05C0\u05C3\u05C6\u05F3\u05F4]", "", text)  # punctuation marks
    return unicodedata.normalize("NFC", text).strip()


def is_hebrew_word(text):
    return bool(re.search(r"[\u05D0-\u05EA]", text)) and len(text) > 1


def sentences_to_dataframe(sentences):
    """Flatten a list of parsed sentences into a single DataFrame with a sentence_num column."""
    rows = []
    for sentence_num, sentence in enumerate(sentences, start=1):
        for token in sentence:
            row = {"sentence_num": sentence_num}
            row.update(token)
            rows.append(row)
    return pd.DataFrame(rows)


def load_corpora():
    """Download, parse, and combine train/dev/test splits for both corpora.

    Returns (ancient_df, modern_df), with ancient word forms stripped of diacritics.
    """
    paths = download_ud_data()

    ancient_sentences = (
        parse_conllu(paths["ancient_train"])
        + parse_conllu(paths["ancient_dev"])
        + parse_conllu(paths["ancient_test"])
    )
    modern_sentences = (
        parse_conllu(paths["modern_train"])
        + parse_conllu(paths["modern_dev"])
        + parse_conllu(paths["modern_test"])
    )

    ancient_df = sentences_to_dataframe(ancient_sentences)
    ancient_df["form"] = ancient_df["form"].apply(remove_hebrew_diacritics)

    modern_df = sentences_to_dataframe(modern_sentences)

    return ancient_df, modern_df


def vocabulary_overlap(ancient_df, modern_df):
    """Report lemma-level vocabulary overlap between the two corpora (diagnostic only)."""
    ancient_words = {
        remove_hebrew_diacritics(lemma)
        for lemma in ancient_df["lemma"]
        if is_hebrew_word(remove_hebrew_diacritics(lemma))
    }
    modern_words = {
        remove_hebrew_diacritics(lemma)
        for lemma in modern_df["lemma"]
        if is_hebrew_word(remove_hebrew_diacritics(lemma))
    }
    shared = ancient_words & modern_words

    print(f"Ancient unique lemmas: {len(ancient_words):,}")
    print(f"Modern unique lemmas:  {len(modern_words):,}")
    print(f"Shared:                {len(shared):,}")
    print(f"Overlap:               {100 * len(shared) / len(ancient_words | modern_words):.1f}%")

    return shared
