"""Prompts for the claim verification pipeline.

This module contains all the LLM prompts used across the verification workflow.
Each prompt is documented with:
  - What it does
  - Why it's designed this way
  - What the LLM is expected to return
  - Example input/output

The prompts are templates — they have {placeholders} that get filled in at
runtime with actual data.


## How Verification Works (the big picture)

The verification pipeline takes a news claim and answers: "Is this true?"

A naive approach would be: send the claim to an LLM and ask "is this true?"
That's bad because:
  1. LLMs hallucinate. They'll confidently say something is true/false
     based on training data that might be wrong or outdated.
  2. No evidence trail. You get an answer but no way to check it.
  3. Complex claims have multiple parts that might each be true/false.

Instead, we break it into steps:

### Step 1: DECOMPOSE
Split a complex claim into simple, atomic sub-claims.

  "The UK spent £50B on HS2 before cancelling the northern leg"
  →  ["The UK spent £50B on HS2",
      "The northern leg of HS2 was cancelled"]

Why? Because each piece might have a different truth value. The UK DID
cancel the northern leg (true), but the exact £50B figure might be wrong
(the actual number was ~£45B at cancellation). Checking them separately
gives a more accurate and useful result.

### Step 2: RESEARCH (for each sub-claim)
Use real tools (web search, Wikipedia, news APIs) to find actual evidence.
This is the LangGraph agent — it searches, reads results, decides if it
has enough, and can loop back to search again.

Why tools instead of asking the LLM directly? Because:
  - Web search gives us current, real-world sources
  - Wikipedia gives us established facts
  - We get actual URLs and excerpts as evidence
  - The LLM's training data has a cutoff date — tools don't

### Step 3: JUDGE (for each sub-claim)
Now the LLM looks at the evidence (not its own knowledge) and judges:
Is this sub-claim supported, contradicted, or unclear?

The key constraint: the LLM must reason ONLY from the evidence provided,
not from its own training data. This is what makes the verdict trustworthy.

### Step 4: SYNTHESIZE
Combine sub-claim verdicts into one overall verdict for the original claim.
If 2 of 3 sub-claims are true and 1 is false, the overall verdict is
"mostly_true" not just "true" or "false".

---

Verdict scale (6 levels):
  - true: All sub-claims are well-supported by evidence
  - mostly_true: Most sub-claims are supported, minor inaccuracies
  - mixed: Some sub-claims true, some false
  - mostly_false: Most sub-claims are contradicted by evidence
  - false: All sub-claims are clearly contradicted
  - unverifiable: Not enough evidence to judge either way
"""


# =============================================================================
# STEP 1: DECOMPOSE
# =============================================================================

DECOMPOSE_SYSTEM = """\
You are a fact-checker's assistant. Your job is to break down complex claims \
into simple, independently verifiable sub-claims.

Rules:
- Each sub-claim must be a single factual assertion
- Each sub-claim must be verifiable on its own (without needing the others)
- Each sub-claim must be specific (include numbers, dates, names when present)
- Do NOT include opinions, predictions, or subjective statements
- Do NOT rephrase the claim — preserve the original wording where possible
- If the claim is already atomic (a single fact), return it as-is in the array

Return ONLY a JSON array of strings. No markdown, no explanation, no wrapping.\
"""

DECOMPOSE_USER = """\
Break this claim into independently verifiable sub-claims.

Claim: {claim_text}

Return a JSON array of strings. Example:
["sub-claim 1", "sub-claim 2"]

/no_think\
"""

# Why /no_think?
#   Qwen3 models support a /no_think token that disables the model's internal
#   chain-of-thought reasoning. For structured output tasks like JSON parsing,
#   we don't want the model to "think out loud" — we just want the JSON array.
#   This reduces latency and avoids the model wrapping its answer in <think>
#   tags.
#
# Why a system + user message split?
#   The system message sets the persona and rules (stable across all calls).
#   The user message provides the specific claim (changes every call).
#   This follows the ChatML format that OpenAI-compatible APIs expect.
#
# Why "Return ONLY a JSON array"?
#   LLMs love to be helpful and add explanations. Without this instruction,
#   you'll get "Sure! Here are the sub-claims: ..." which breaks JSON parsing.
#
# Example input/output:
#   Input:  "NASA spent $25.4 billion on the Apollo program and landed
#            humans on the moon 6 times between 1969 and 1972"
#   Output: ["NASA spent $25.4 billion on the Apollo program",
#            "NASA landed humans on the moon 6 times",
#            "The moon landings occurred between 1969 and 1972"]


# =============================================================================
# STEP 2: RESEARCH (agent system prompt)
# =============================================================================

RESEARCH_SYSTEM = """\
You are a research assistant tasked with gathering evidence about a specific \
factual claim. You have access to web search and Wikipedia tools.

Your goal: find 2-5 pieces of evidence from real sources that either SUPPORT \
or CONTRADICT the claim. Quality over quantity.

Research strategy:
1. Search for the claim directly — look for fact-check articles or news reports
2. Search for the specific entities, numbers, or dates in the claim
3. Check Wikipedia for established background facts
4. If initial results are thin, try different phrasings or related terms

When to stop:
- You have found at least 2-3 relevant sources from different searches
- OR you have done 3-4 searches and found nothing relevant (the claim \
  may be unverifiable)

Do NOT make up evidence. Only report what the tools actually return.
Do NOT evaluate whether the claim is true or false — just gather evidence.

When you have finished searching, write a brief summary of what you found.\
"""

RESEARCH_USER = """\
Find evidence about this claim:

"{sub_claim}"

Identify the KEY DETAIL that makes this claim specific and verifiable, then \
search for THAT. Don't just search for the people or topic in general — \
search for the specific event, action, number, or object mentioned.

Use both web search and Wikipedia when relevant.\
"""

# Why separate RESEARCH from JUDGE?
#   Research and judgment are different skills. Research = "find relevant
#   information." Judgment = "evaluate what that information means."
#   Splitting them means:
#     1. The research agent can focus purely on finding good sources
#     2. The judge can focus purely on evaluation
#     3. We can swap out research strategies without affecting judgment
#     4. Evidence is collected independently — the judge sees ALL of it,
#        not a cherry-picked subset
#
# Why "Do NOT evaluate whether the claim is true or false"?
#   If the research agent judges while searching, it might stop early
#   ("I found one source saying it's true, done!"). We want it to keep
#   searching for contradicting evidence too.
#
# Why limit to 3-4 searches?
#   Each search costs time and LLM tokens. More searches ≠ better evidence.
#   3-4 well-targeted searches usually find what's out there. If nothing
#   comes up in 4 searches, more won't help — the claim is likely too
#   obscure to verify.


# =============================================================================
# STEP 3: JUDGE
# =============================================================================

JUDGE_SYSTEM = """\
You are an impartial fact-checker. You will be given a claim and a set of \
evidence gathered from real sources (web search, Wikipedia, news articles).

Your job:
1. Evaluate each piece of evidence — does it SUPPORT, CONTRADICT, or say \
nothing about the claim?
2. Weigh the evidence — reliable sources (official reports, major news \
outlets, academic sources) count more than blogs or social media.
3. Render a verdict based ONLY on the evidence provided. Do NOT use your \
own knowledge.

Verdict scale:
- "true" — evidence clearly supports the claim
- "false" — evidence clearly contradicts the claim
- "partially_true" — claim is broadly correct but has inaccuracies \
(wrong numbers, missing context, etc.)
- "unverifiable" — not enough evidence to judge either way

Confidence scoring (USE THE FULL RANGE — do NOT default to 0.9+):
- 0.95-1.0 — Multiple high-quality sources explicitly confirm/deny. No \
ambiguity whatsoever. Reserve this for slam-dunk cases only.
- 0.80-0.94 — Strong evidence from reliable sources, but minor gaps \
(e.g., exact figures differ slightly, or only 1-2 strong sources).
- 0.60-0.79 — Moderate evidence. Sources partially address the claim, or \
there are conflicting signals between sources.
- 0.40-0.59 — Weak evidence. Sources are tangential, low-quality, or \
contradict each other roughly equally.
- 0.20-0.39 — Very little relevant evidence found. Verdict is mostly a \
best guess.
- 0.0-0.19 — Essentially no usable evidence. Almost pure uncertainty.

Be calibrated: if the evidence is decent but not overwhelming, use 0.7 \
or 0.75 — not 0.95. Only use 0.9+ when the evidence is rock-solid from \
multiple authoritative sources.

Return a JSON object:
{
  "verdict": "true|false|partially_true|unverifiable",
  "confidence": 0.0 to 1.0,
  "reasoning": "Brief explanation of how the evidence supports your verdict"
}

Return ONLY the JSON object. No markdown, no explanation, no wrapping.\
"""

JUDGE_USER = """\
Judge this claim based ONLY on the evidence below. Do not use your own knowledge.

Claim: {sub_claim}

Evidence:
{evidence_text}

Think carefully about what the evidence says. Weigh conflicting sources. \
Then return a JSON object with "verdict", "confidence", and "reasoning".\
"""

# Why "Do NOT use your own knowledge"?
#   This is the critical constraint. Without it, the LLM will just answer from
#   memory, which defeats the entire purpose of gathering evidence. We want
#   the verdict to be grounded in real, citable sources — not the model's
#   training data (which may be wrong or outdated).
#
# Why confidence as a float?
#   0.0 = "I have no idea" to 1.0 = "absolutely certain". This lets us
#   distinguish between "true with high confidence" (strong evidence from
#   multiple sources) and "true with low confidence" (one weak source).
#   The frontend can use this to highlight uncertain verdicts.
#
# Example input/output:
#   Input claim: "NASA spent $25.4 billion on the Apollo program"
#   Input evidence: [
#     {source: "Wikipedia", content: "The Apollo program cost $25.4B..."},
#     {source: "NASA.gov", content: "Total program cost: $25.4 billion..."}
#   ]
#   Output: {
#     "verdict": "true",
#     "confidence": 0.85,
#     "reasoning": "Two reliable sources (Wikipedia, NASA.gov) confirm the
#                   $25.4 billion figure. Confidence is 0.85 rather than
#                   higher because the exact figure may vary depending on
#                   whether inflation-adjusted."
#   }


# =============================================================================
# STEP 4: SYNTHESIZE
# =============================================================================

SYNTHESIZE_SYSTEM = """\
You are an impartial fact-checker. You have received verdicts for each \
sub-claim that makes up a larger claim. Your job is to combine them into \
a single overall verdict.

Rules:
- If ALL sub-claims are true → "true"
- If MOST are true with minor issues → "mostly_true"
- If roughly half true and half false → "mixed"
- If MOST are false → "mostly_false"
- If ALL sub-claims are false → "false"
- If most evidence is insufficient → "unverifiable"

The overall confidence should reflect the weakest link — if one sub-claim \
is very uncertain, your overall confidence should be lower.

Confidence scoring (USE THE FULL RANGE):
- 0.95-1.0 — All sub-claims have rock-solid verdicts. Reserve for slam-dunks.
- 0.80-0.94 — Strong but not perfect. Most sub-claims well-supported.
- 0.60-0.79 — Moderate. Some sub-claims uncertain or evidence is mixed.
- 0.40-0.59 — Weak. Significant uncertainty in multiple sub-claims.
- Below 0.40 — Very uncertain. Mostly guesswork.

Do NOT default to 0.9+. Be honest about uncertainty.

Return a JSON object:
{
  "verdict": "true|mostly_true|mixed|mostly_false|false|unverifiable",
  "confidence": 0.0 to 1.0,
  "reasoning": "Brief summary of how the sub-verdicts combine"
}

Return ONLY the JSON object. No markdown, no explanation, no wrapping.\
"""

SYNTHESIZE_USER = """\
Combine these sub-claim verdicts into an overall verdict for the original claim.

Original claim: {claim_text}

Sub-claim verdicts:
{sub_verdicts_text}

Return a JSON object with "verdict", "confidence", and "reasoning".

/no_think\
"""

# Why synthesize with an LLM instead of just averaging?
#   Because verdict logic isn't simple math. If someone claims "X happened
#   in 2019 and cost $50M" and the event DID happen but in 2020 and cost
#   $48M, a human would say "mostly true" — the core claim is right, the
#   details are slightly off. An LLM can make this nuanced judgment better
#   than a simple "2 of 3 sub-claims are true → 66% → mostly_true".
