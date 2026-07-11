"""
Stage 3: train grammatical embeddings (POS, head-POS, dependency relation)
with contrastive triplet loss, so that words sharing a grammatical profile end
up close together in the combined embedding space, regardless of which corpus
they came from.

The AlephBERT word embeddings themselves are frozen; only the grammar
embedding tables are updated.
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


def build_index_mappings(combined_df):
    """Map each grammatical tag to a fixed integer index, and build the AlephBERT
    embedding tensor plus tag-index tensors aligned to combined_df's row order.
    """
    pos_tags = sorted(combined_df["upos"].unique())
    head_pos_tags = sorted(combined_df["head_pos"].unique())
    dep_rels = sorted(combined_df["deprel"].unique())

    pos_to_idx = {tag: i for i, tag in enumerate(pos_tags)}
    head_pos_to_idx = {tag: i for i, tag in enumerate(head_pos_tags)}
    dep_to_idx = {rel: i for i, rel in enumerate(dep_rels)}

    alephbert_embs = torch.tensor(np.vstack(combined_df["embedding"].values), dtype=torch.float32)
    pos_indices = torch.tensor([pos_to_idx[t] for t in combined_df["upos"]], dtype=torch.long)
    head_pos_indices = torch.tensor(
        [head_pos_to_idx[t] for t in combined_df["head_pos"]], dtype=torch.long
    )
    dep_indices = torch.tensor([dep_to_idx[t] for t in combined_df["deprel"]], dtype=torch.long)

    return {
        "pos_to_idx": pos_to_idx,
        "head_pos_to_idx": head_pos_to_idx,
        "dep_to_idx": dep_to_idx,
        "alephbert_embs": alephbert_embs,
        "pos_indices": pos_indices,
        "head_pos_indices": head_pos_indices,
        "dep_indices": dep_indices,
    }


class GrammarEmbeddings:
    """Holds the three trainable grammar embedding tables and combines them
    with a (frozen) AlephBERT word embedding.
    """

    def __init__(self, device):
        self.pos_embeddings = nn.Embedding(config.NUM_POS_TAGS, config.POS_EMBED_DIM)
        self.head_pos_embeddings = nn.Embedding(config.NUM_HEAD_POS_TAGS, config.HEAD_POS_EMBED_DIM)
        self.dep_embeddings = nn.Embedding(config.NUM_DEP_RELS, config.DEP_EMBED_DIM)

        for emb in (self.pos_embeddings, self.head_pos_embeddings, self.dep_embeddings):
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)
            emb.to(device)

    def parameters(self):
        return (
            list(self.pos_embeddings.parameters())
            + list(self.head_pos_embeddings.parameters())
            + list(self.dep_embeddings.parameters())
        )

    def combine(self, alephbert, pos_idx, head_pos_idx, dep_idx):
        return torch.cat(
            [
                alephbert,
                self.pos_embeddings(pos_idx),
                self.head_pos_embeddings(head_pos_idx),
                self.dep_embeddings(dep_idx),
            ],
            dim=1,
        )

    def as_numpy_dict(self, pos_to_idx, head_pos_to_idx, dep_to_idx):
        return {
            "pos_embeddings": self.pos_embeddings.weight.detach().cpu().numpy(),
            "head_pos_embeddings": self.head_pos_embeddings.weight.detach().cpu().numpy(),
            "dep_embeddings": self.dep_embeddings.weight.detach().cpu().numpy(),
            "pos_to_idx": pos_to_idx,
            "head_pos_to_idx": head_pos_to_idx,
            "dep_to_idx": dep_to_idx,
        }


def train_grammar_embeddings(combined_df, triplets, seed=config.RANDOM_SEED):
    """Train the grammar embeddings with triplet margin loss over the given
    (anchor, positive, negative) index triplets. Returns the trained embeddings
    packaged as a plain dict, ready to pickle.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    mappings = build_index_mappings(combined_df)
    grammar = GrammarEmbeddings(device)

    triplet_criterion = nn.TripletMarginLoss(margin=config.TRIPLET_MARGIN, p=2, eps=1e-8)
    optimizer = torch.optim.Adam(grammar.parameters(), lr=config.LEARNING_RATE)

    def full_embedding(indices):
        alephbert = mappings["alephbert_embs"][indices].to(device)
        pos = mappings["pos_indices"][indices].to(device)
        head_pos = mappings["head_pos_indices"][indices].to(device)
        dep = mappings["dep_indices"][indices].to(device)
        return grammar.combine(alephbert, pos, head_pos, dep)

    triplets = list(triplets)

    for epoch in range(config.EPOCHS):
        random.shuffle(triplets)
        total_loss = 0.0
        num_batches = 0

        for start in range(0, len(triplets), config.BATCH_SIZE):
            batch = triplets[start : start + config.BATCH_SIZE]
            anchor_idx = [t[0] for t in batch]
            pos_idx = [t[1] for t in batch]
            neg_idx = [t[2] for t in batch]

            anchor_emb = F.normalize(full_embedding(anchor_idx), p=2, dim=1)
            positive_emb = F.normalize(full_embedding(pos_idx), p=2, dim=1)
            negative_emb = F.normalize(full_embedding(neg_idx), p=2, dim=1)

            loss = triplet_criterion(anchor_emb, positive_emb, negative_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        print(f"Epoch {epoch + 1}/{config.EPOCHS} | Avg Loss: {total_loss / num_batches:.4f}")

    print("Training complete.")

    return grammar.as_numpy_dict(
        mappings["pos_to_idx"], mappings["head_pos_to_idx"], mappings["dep_to_idx"]
    )
