"""Deterministic embedding-based sentence grouping.

Groups consecutive same-speaker sentences by cosine similarity of their
embeddings. Speaker changes always start a new group. A bridge heuristic
keeps short vague sentences (e.g. "It was quite something") in the group
when the sentences on either side are topically similar.

Returns SentenceGroup dataclasses — downstream synthesis uses these to
batch sentences for claim writing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SentenceGroup:
    """A group of consecutive same-speaker sentences on one topic."""
    group_id: int
    sentence_indices: list[int] = field(default_factory=list)
    speaker: str = ""


def group_sentences(
    sentences: list[dict],
    embeddings: np.ndarray,
    threshold: float,
    bridge_max_words: int,
) -> list[SentenceGroup]:
    """Group sentences by speaker boundaries + embedding similarity.

    Args:
        sentences: Ordered by global_index. Keys: global_index, speaker, standalone.
        embeddings: Shape (N, dim), L2-normalized, same order as sentences.
        threshold: Cosine similarity threshold for topic continuity.
        bridge_max_words: Max words for a sentence to be eligible for bridging.

    Returns:
        List of SentenceGroup in sentence order.
    """
    if not sentences:
        return []

    n = len(sentences)
    assert embeddings.shape[0] == n, (
        f"Embedding count {embeddings.shape[0]} != sentence count {n}"
    )

    groups: list[SentenceGroup] = []
    current_group = SentenceGroup(
        group_id=1,
        sentence_indices=[sentences[0]["global_index"]],
        speaker=sentences[0]["speaker"],
    )
    # Running centroid for the current group (start with first sentence's embedding)
    centroid = embeddings[0].copy()
    centroid_count = 1

    for i in range(1, n):
        sent = sentences[i]
        speaker = sent["speaker"]
        emb = embeddings[i]

        # Speaker change → always new group
        if speaker != current_group.speaker:
            groups.append(current_group)
            current_group = SentenceGroup(
                group_id=len(groups) + 2,
                sentence_indices=[sent["global_index"]],
                speaker=speaker,
            )
            centroid = emb.copy()
            centroid_count = 1
            continue

        # Same speaker — check cosine similarity to group centroid
        sim = float(np.dot(emb, centroid))

        if sim >= threshold:
            # Topically similar → add to group, update centroid
            current_group.sentence_indices.append(sent["global_index"])
            centroid_count += 1
            centroid = centroid + (emb - centroid) / centroid_count
            # Re-normalize for future dot-product comparisons
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            continue

        # Below threshold — try bridge heuristic for short sentences
        word_count = len(sent["standalone"].split())
        if (
            word_count <= bridge_max_words
            and i + 1 < n
            and sentences[i + 1]["speaker"] == current_group.speaker
        ):
            # Check if the NEXT same-speaker sentence is similar to centroid
            next_sim = float(np.dot(embeddings[i + 1], centroid))
            if next_sim >= threshold:
                # Bridge: keep the short sentence in the group
                current_group.sentence_indices.append(sent["global_index"])
                centroid_count += 1
                centroid = centroid + (emb - centroid) / centroid_count
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
                continue

        # New group
        groups.append(current_group)
        current_group = SentenceGroup(
            group_id=len(groups) + 2,
            sentence_indices=[sent["global_index"]],
            speaker=speaker,
        )
        centroid = emb.copy()
        centroid_count = 1

    # Don't forget the last group
    groups.append(current_group)

    return groups
