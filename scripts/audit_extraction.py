"""Audit extraction coverage: sentencize transcript, map each sentence to claims."""
import sys
sys.path.insert(0, "/app")

from src.transcript.thesis_extractor import sentencize_transcript
from src.transcript.parsers import SpeakerTurn

import psycopg2
conn = psycopg2.connect(host="spin-cycle-dev-postgres", dbname="spincycle", user="scuser", password="scpass")
cur = conn.cursor()

TID = "fd599bec-c4c4-42d1-9929-9c67ac97cd51"

cur.execute("SELECT display_text, segments_data FROM transcripts WHERE id = %s", (TID,))
display_text, segments_data = cur.fetchone()

turns = []
for seg in segments_data:
    turns.append(SpeakerTurn(
        speaker=seg["speaker"],
        text=seg["text"],
        section_header=seg.get("section_header"),
    ))

sentences = sentencize_transcript(turns)
print(f"Total sentences: {len(sentences)}")

cur.execute("""
    SELECT claim_text, original_quote, is_duplicate, dedup_group_id, classification, checkable
    FROM transcript_claims
    WHERE transcript_id = %s
    ORDER BY created_at
""", (TID,))
claims = cur.fetchall()
print(f"Total claims: {len(claims)}")

# Map sentences to claims via original_quote substring matching
sentence_to_claims = {}
uncovered = []

for i, sent in enumerate(sentences):
    matched = False
    for claim_text, orig_quote, is_dup, dedup_group, classification, checkable in claims:
        if sent.text.strip() in orig_quote:
            sentence_to_claims.setdefault(i, []).append({
                "claim": claim_text[:120],
                "group": dedup_group,
                "is_dup": is_dup,
            })
            matched = True
    if not matched:
        uncovered.append(i)

print(f"\nCovered sentences: {len(sentences) - len(uncovered)}")
print(f"Uncovered sentences (not_claims): {len(uncovered)}")

SEP = "=" * 80

print(f"\n{SEP}")
print("UNCOVERED SENTENCES (marked not_claim):")
print(SEP)
for idx in uncovered:
    s = sentences[idx]
    print(f"  [S{idx}] {s.speaker}: {s.text}")

# Build sentence -> primary claim group
sent_group = {}
for i in range(len(sentences)):
    if i in sentence_to_claims:
        groups = [c["group"] for c in sentence_to_claims[i] if not c["is_dup"]]
        if groups:
            sent_group[i] = groups[0]

# Find consecutive claim sentences in different groups
split_points = []
prev_idx = None
prev_group = None
for idx in sorted(sent_group.keys()):
    if prev_idx is not None and idx == prev_idx + 1 and sent_group[idx] != prev_group:
        split_points.append((prev_idx, idx, prev_group, sent_group[idx]))
    prev_idx = idx
    prev_group = sent_group[idx]

print(f"\n{SEP}")
print(f"GROUP BOUNDARIES ({len(split_points)} transitions between consecutive claim sentences):")
print(SEP)
for s1, s2, g1, g2 in split_points:
    sent1 = sentences[s1]
    sent2 = sentences[s2]
    c1 = sentence_to_claims[s1][0]["claim"][:100]
    c2 = sentence_to_claims[s2][0]["claim"][:100]
    print(f"  [S{s1}] {sent1.text[:120]}")
    print(f"    -> claim: {c1}")
    print(f"  [S{s2}] {sent2.text[:120]}")
    print(f"    -> claim: {c2}")
    print()

# Summary stats
print(f"\n{SEP}")
print("SUMMARY")
print(SEP)
dup_count = sum(1 for c in claims if c[2])  # is_duplicate
not_checkable = sum(1 for c in claims if c[4] == "not_checkable")
print(f"  Sentences: {len(sentences)}")
print(f"  Covered (claim): {len(sentences) - len(uncovered)}")
print(f"  Uncovered (not_claim): {len(uncovered)}")
print(f"  Claims extracted: {len(claims)}")
print(f"  Duplicates flagged: {dup_count}")
print(f"  Not checkable: {not_checkable}")
print(f"  Unique checkable: {len(claims) - dup_count - not_checkable}")

conn.close()
