# Bridging Ancient and Modern Hebrew Without Parallel Text

Can grammatical structure alone help translate between two forms of a language that have no
parallel corpus connecting them?

This project trains a translation approach between Ancient Hebrew (Biblical Hebrew) and Modern
Hebrew using only independently-collected, non-parallel treebanks, no aligned ancient-modern
sentence pairs exist for this language pair, unlike most translation research, which relies on
parallel text almost by default. The core idea: use each word's grammatical role (part of speech,
dependency relation, the part of speech of the word it depends on) as a bridge between the two
corpora, learned through contrastive embedding training.

## Background

This started as an attempt to combine an interest in historical linguistics with NLP outside of
English. Hebrew seemed like a natural fit, ancient and modern Hebrew are close enough that native
speakers today can read Biblical Hebrew without translation, which suggested there might be enough
underlying structural continuity to exploit computationally, even without a parallel corpus to
train on directly.

## Approach

1. **Extract contextual embeddings** for every word in both corpora using
   [AlephBERT](https://huggingface.co/onlplab/alephbert-base), a Hebrew BERT model, with each
   word's embedding computed in its full sentence context (`src/embed.py`).
2. **Build contrastive triplets** across both corpora: two words are a "positive" pair if they
   share the same part-of-speech tag, dependency relation, and head word's part of speech;
   otherwise they're a "negative" pair (`src/triplets.py`). Triplets are constructed three ways
   (ancient-anchor/modern-positive, ancient-anchor/ancient-positive, and a third variant) to
   maximize how much of the ancient corpus contributes to training.
3. **Train grammar embeddings** with triplet margin loss, so that words with matching grammatical
   roles end up close together in embedding space, regardless of which corpus they're from
   (`src/train.py`). The AlephBERT embeddings themselves stay frozen; only the added grammar
   embedding layers are trained.
4. **Translate and evaluate** by finding, for each ancient word, its nearest modern-Hebrew
   neighbor under three different embedding configurations, then re-embedding the resulting
   "translated" sentences and measuring cosine similarity back to the originals (`src/translate.py`).

## Data

- **Ancient Hebrew:** [UD_Ancient_Hebrew-PTNK](https://github.com/UniversalDependencies/UD_Ancient_Hebrew-PTNK) — the Torah and the Book of Ruth (~5,600 sentences)
- **Modern Hebrew:** [UD_Hebrew-HTB](https://github.com/UniversalDependencies/UD_Hebrew-HTB) — Ha'aretz newspaper text (~6,200 sentences)

Both are pulled automatically by `src/data.py`, no manual download needed.

**A caveat worth stating plainly:** the ancient corpus is entirely religious text, and the modern
corpus comes from a single newspaper with a known political leaning. Neither is representative of
the full range of ancient or modern Hebrew, and this project should not be read as producing a
general-purpose translation system, see Limitations below.

## Results

Translation quality was measured as average cosine similarity between each ancient word's original
embedding and its translated modern counterpart's embedding, across a 20% sample of the ancient
corpus, higher is better.

| Configuration | Average Similarity | Change from Word-only |
|---|---|---|
| Word-only | 0.393 | — |
| Word + POS | 0.439 | **+12%** |
| Word + POS + Head-POS + Dependency | 0.372 | −5% |

Adding part-of-speech information measurably improved translation quality. Adding the *full*
grammatical feature set (POS + head-POS + dependency relation) did not continue that trend, it
performed worse than the word-only baseline, not just worse than the POS-only configuration.

Breaking results down by part of speech shows the effect isn't uniform:

| POS | Count | Word-only | Word + POS | Full features | Best config |
|---|---|---|---|---|---|
| NOUN | 6,185 | 0.398 | 0.428 | 0.357 | Word + POS |
| VERB | 3,450 | 0.379 | 0.426 | 0.343 | Word + POS |
| ADJ | 275 | 0.397 | 0.434 | 0.346 | Word + POS |
| NUM | 479 | 0.360 | 0.503 | 0.371 | Word + POS |
| ADP | 4,160 | 0.391 | 0.458 | 0.388 | Word + POS |
| CCONJ | 2,842 | 0.531 | 0.591 | 0.519 | Word + POS |
| ADV | 631 | 0.346 | 0.490 | 0.401 | Word + POS |
| DET | 1,727 | 0.412 | 0.495 | 0.447 | Word + POS |
| PRON | 2,894 | 0.388 | 0.359 | 0.316 | **Word-only** |
| PUNCT | 2,581 | 0.288 | 0.323 | 0.287 | Word + POS |

Numbers (NUM) benefited the most from adding POS information (+40% over word-only), the largest
swing of any category. Pronouns (PRON) were the one part of speech where adding grammatical
features actively hurt, word-only embeddings translated them best. That split suggests how much
grammatical context helps depends on the word type, more features isn't automatically better.

Run `python run_pipeline.py` to reproduce these results, or `python run_pipeline.py --stage train`
to stop after training and inspect the embeddings directly.

## Limitations

- AlephBERT itself is trained only on modern Hebrew, so its embeddings of ancient text may carry
  some bias or reduced quality.
- The grammar embedding dimensions were chosen arbitrarily rather than tuned.
- Translation is done via simple nearest-neighbor cosine similarity; a method that translates full
  sentences jointly, rather than word by word, would likely perform better.
- Negative examples in the triplets share *no* grammatical features with the anchor; using
  negatives that partially overlap might give the contrastive model richer signal to learn from.

## On responsible use

This project is a research exploration of whether grammatical structure can substitute for
parallel text, not a translation tool. Historical translation, particularly of religious texts
like the Hebrew Bible, has real stakes: poor or agenda-driven translations have been used to
justify harm throughout history. This system was never intended to produce authoritative
translations of scripture, and shouldn't be used for that purpose.

## Setup

```bash
pip install -r requirements.txt
python run_pipeline.py
```

## Project structure

```
src/
  config.py       # paths, URLs, hyperparameters — single source of truth
  utils.py        # load_or_compute: shared caching helper
  data.py         # download + parse the UD treebanks
  embed.py        # AlephBERT contextual word embeddings
  triplets.py     # contrastive triplet construction
  train.py        # grammar embedding training (triplet loss)
  translate.py    # nearest-neighbor translation + evaluation
run_pipeline.py    # CLI entry point
```

## License

MIT.
