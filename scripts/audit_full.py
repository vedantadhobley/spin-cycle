"""Full audit: every sentence → claim mapping + decontextualized text."""
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

cur.execute("""
    SELECT claim_text, original_quote
    FROM transcript_claims
    WHERE transcript_id = %s
    ORDER BY created_at
""", (TID,))
claims = cur.fetchall()

# Build claim index: map each claim to its ordered number
claim_list = []
for i, (claim_text, orig_quote) in enumerate(claims):
    claim_list.append({
        "num": i + 1,
        "text": claim_text,
        "quote": orig_quote,
    })

# Map each sentence to its claim(s) via original_quote substring matching
sent_claims = {}
for i, sent in enumerate(sentences):
    for c in claim_list:
        if sent.text.strip() in c["quote"]:
            sent_claims.setdefault(i, []).append(c["num"])

# Group sentences by claim number
claim_sentences = {}
for idx, cnums in sent_claims.items():
    for cn in cnums:
        claim_sentences.setdefault(cn, []).append(idx)

SEP = "=" * 100

# Print full mapping: for each claim, show sentences then decontextualized text
print(f"TRANSCRIPT: {len(sentences)} sentences → {len(claims)} claims, {len(sentences) - len(sent_claims)} not_claim\n")

for c in claim_list:
    idxs = claim_sentences.get(c["num"], [])
    idx_str = ", ".join(f"S{i}" for i in sorted(idxs))
    print(SEP)
    print(f"CLAIM {c['num']:>2} ({len(idxs)} sentences: {idx_str})")
    print(SEP)
    print()
    # Show each sentence
    for idx in sorted(idxs):
        s = sentences[idx]
        print(f"  [S{idx:>3}] {s.text}")
    print()
    print(f"  DECONTEXTUALIZED:")
    # Word wrap the claim text at ~100 chars
    words = c["text"].split()
    line = "    "
    for w in words:
        if len(line) + len(w) + 1 > 100:
            print(line)
            line = "    " + w
        else:
            line += " " + w if line.strip() else "    " + w
    if line.strip():
        print(line)
    print()

# Show not_claim sentences
print(SEP)
print("NOT_CLAIM SENTENCES")
print(SEP)
not_claim_idxs = [i for i in range(len(sentences)) if i not in sent_claims]
for idx in not_claim_idxs:
    s = sentences[idx]
    print(f"  [S{idx:>3}] {s.text}")
print()

# Check for potential issues
print(SEP)
print("POTENTIAL ISSUES")
print(SEP)
issues = 0

# 1. Very short decontextualized statements (< 50 chars)
for c in claim_list:
    if len(c["text"]) < 50:
        issues += 1
        print(f"  SHORT CLAIM {c['num']}: ({len(c['text'])} chars) {c['text']}")

# 2. Single-sentence claims from short sentences (< 10 words)
for c in claim_list:
    idxs = claim_sentences.get(c["num"], [])
    if len(idxs) == 1:
        s = sentences[idxs[0]]
        words = len(s.text.split())
        if words < 10:
            issues += 1
            print(f"  SHORT SOLO S{idxs[0]} ({words} words): {s.text}")
            print(f"    → CLAIM {c['num']}: {c['text'][:100]}")

# 3. Claims that seem like pure rhetoric / not checkable
rhetoric_markers = ["unstoppable", "extraordinary", "brave", "beautiful", "amazing", "incredible"]
for c in claim_list:
    text_lower = c["text"].lower()
    if any(m in text_lower for m in rhetoric_markers):
        idxs = claim_sentences.get(c["num"], [])
        idx_str = ", ".join(f"S{i}" for i in sorted(idxs))
        issues += 1
        print(f"  RHETORIC? CLAIM {c['num']} ({idx_str}): {c['text'][:120]}")

if issues == 0:
    print("  None found.")
print(f"\nTotal potential issues: {issues}")

conn.close()
