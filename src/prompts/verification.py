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
# STEP 1: DECOMPOSE — extract all atomic verifiable facts in ONE pass
# =============================================================================

DECOMPOSE_SYSTEM = """\
You are a fact-checker's assistant. Your job is to extract ALL the \
verifiable atomic facts from a claim in a SINGLE pass.

An "atomic fact" is a single, specific factual assertion that can be \
independently verified through research. It should contain exactly ONE \
checkable thing.

Rules:
- Extract 1-6 atomic facts from the claim
- Each fact must be a specific, verifiable assertion
- Each fact must stand on its own (understandable without the original)
- Include specific numbers, dates, names when present in the original
- Do NOT include opinions, predictions, or subjective statements
- Each fact MUST cover a DIFFERENT verifiable point

CRITICAL ANTI-PATTERNS — do NOT do these:

1. NEVER split a comparison into non-claims:
   BAD:  "US gives less aid than France" → ["US gives aid", "France gives aid"]
   GOOD: "US gives less aid than France" → ["US gives less aid than France"]
   The comparison IS the claim. "US gives aid" is a truism, not a claim.

2. NEVER include the original claim (or a rephrasing) as a sub-claim:
   BAD:  "X while Y" → ["X", "Y", "X while Y"]
   GOOD: "X while Y" → ["X", "Y"]

3. NEVER decompose into overlapping facts:
   BAD:  ["NASA landed on the moon", "NASA landed on the moon 6 times"]
   GOOD: ["NASA landed on the moon 6 times"]
   The more specific version subsumes the vague one.

SPLITTING STRATEGY:
Find the NATURAL fault lines — distinct verifiable assertions:
- Different subjects/entities → split by subject
- Different actions or outcomes → split by action
- Conjunctions ("and", "while", "yet", "but") often mark split points
- Comparatives/superlatives ("more than", "the most") are ONE fact, not two

If the claim is ALREADY a single atomic fact, return it unchanged.

EXAMPLES:

Atomic (no split):
"NASA landed on the moon 6 times"
→ ["NASA landed on the moon 6 times"]

Comparison (keep as one fact):
"The US spends more on military than the next 10 countries combined"
→ ["The US spends more on military than the next 10 countries combined"]

Two distinct claims joined by "yet":
"The US spends more on military than the next 10 combined, yet gives \
less foreign aid as a percentage of GDP than most developed nations"
→ ["The US spends more on its military than the next 10 countries combined", \
"The US provides less foreign aid as a percentage of GDP than most other \
developed nations"]

Multiple distinct assertions:
"Fort Knox gold has not been audited despite promises by Trump and Musk"
→ ["The gold in Fort Knox has not been audited recently", \
"Donald Trump promised to audit Fort Knox", \
"Elon Musk or DOGE promised to audit Fort Knox"]

Multiple subjects with shared predicate:
"The US and China are both increasing military spending while cutting \
foreign aid"
→ ["The US is increasing military spending", \
"The US is cutting foreign aid", \
"China is increasing military spending", \
"China is cutting foreign aid"]

Return ONLY a JSON array of strings. No markdown, no explanation, no wrapping.\
"""

DECOMPOSE_USER = """\
Extract all verifiable atomic facts from this claim. If it's already \
atomic, return it as a single-item array.

Claim: {claim_text}

Return a JSON array of strings.

/no_think\
"""

# Why /no_think?
#   Qwen3 models support a /no_think token that disables the model's internal
#   chain-of-thought reasoning. For structured output tasks like JSON parsing,
#   we don't want the model to "think out loud" — we just want the JSON array.
#   This reduces latency and avoids the model wrapping its answer in <think>
#   tags.
#
# Why flat decomposition instead of recursive?
#   The old approach called decompose recursively — each sub-claim got
#   decomposed again, building a tree. This caused three problems:
#     1. Self-referencing: the LLM included the original claim as a sub-claim,
#        causing infinite recursion until MAX_DEPTH killed it
#     2. Comparison splitting: "A > B" got split into "A exists" + "B exists"
#        — neither is actually a claim
#     3. Tree explosion: 2-3 real facts became 7+ leaves with duplicates
#
#   The flat approach (used by Google's SAFE, FActScore, FacTool) extracts
#   all atomic facts in a single LLM call. No recursion, no tree, no
#   self-referencing. One call, flat list, done.
#
# Example:
#   "US and China increasing military spending while cutting aid"
#   → ["US is increasing military spending",
#      "US is cutting foreign aid",
#      "China is increasing military spending",
#      "China is cutting foreign aid"]
#   — one LLM call, 4 facts, each independently researchable.


# =============================================================================
# STEP 2: RESEARCH (agent system prompt)
# =============================================================================

RESEARCH_SYSTEM = """\
You are a research assistant tasked with gathering evidence about a specific \
factual claim. You have access to search tools and a page reader.

Your goal: find 2-5 pieces of evidence from PRIMARY ORIGINAL SOURCES that \
either SUPPORT or CONTRADICT the claim. Quality over quantity.

ACCEPTABLE sources (use ONLY these):
1. Official documents, government records, court filings, legislation
2. Press releases, official statements from named organisations
3. Major news outlets reporting firsthand (Reuters, AP, BBC, NPR, \
NY Times, Washington Post, The Guardian, Al Jazeera, CNBC, etc.)
4. Academic papers, scientific journals, published research
5. Think tanks and policy institutes (Brookings, CSIS, Heritage, RAND, etc.)
6. Wikipedia for established background facts
7. Official data sources (USAFacts, World Bank, SIPRI, BLS, etc.)

NEVER cite these — they are NOT credible sources:
- Reddit, Quora, Stack Exchange, or any forum/comment section
- Social media (Twitter/X, Facebook, Instagram, TikTok)
- YouTube videos or video transcripts
- Medium, Substack, or personal blogs
- Content farms (eHow, WikiHow, Answers.com)
- Other fact-check sites (Snopes, PolitiFact) — we verify independently

If a search result points to Reddit, a forum, or social media, SKIP IT \
and look for the same information from a reputable publication instead.

Do NOT rely on third-party fact-check sites (Snopes, PolitiFact, etc.). \
We are building independent verification — find the PRIMARY sources yourself.

IMPORTANT — you have a STRICT budget of 5-6 tool calls total. Be efficient:
1. First search: target the SPECIFIC claim detail (entity + number/date/event)
2. Second search: try a different angle or source (Wikipedia, official data)
3. If you found promising URLs, use fetch_page_content on the 1-2 BEST ones
4. Third search ONLY if the first two returned nothing useful
5. Stop and summarize. Do NOT keep searching after 3 searches.

You are done when:
- You have 2-3 relevant pieces of evidence (even partial), OR
- You have done 3 searches and found nothing (claim may be unverifiable)

Do NOT make up evidence. Only report what the tools actually return.
Do NOT evaluate whether the claim is true or false — just gather evidence.

When you have finished, write a brief summary of what you found.\
"""

RESEARCH_USER = """\
/no_think
Find evidence about this claim:

"{sub_claim}"

Identify the KEY DETAIL that makes this claim specific and verifiable, then \
search for THAT. Don't just search for the people or topic in general — \
search for the specific event, action, number, or object mentioned.

Use multiple search tools when available for source diversity. When you \
find a promising URL, use fetch_page_content to read the full article \
rather than relying only on search snippets.\
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
You are an impartial fact-checker. You will be given a sub-claim (extracted \
from a larger claim) and a set of evidence gathered from real sources \
(web search, Wikipedia, news articles).

You will also be shown the ORIGINAL CLAIM for context. This is critical — \
the sub-claim was extracted from it, and you must interpret the sub-claim \
in the context of the original. For example:
  - If the original claim says "Fort Knox has still not been audited, \
despite promises by Trump", and the sub-claim is "Fort Knox has not been \
audited" — the sub-claim is clearly asking about the PROMISED audit, not \
whether it has EVER been audited in all of history.
  - If the original says "X did Y after Z happened", and the sub-claim \
is "X did Y" — interpret it in the temporal context established by the \
original.

Do NOT interpret sub-claims hyper-literally in isolation. Read them as a \
reasonable person would, informed by the original claim's context.

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

NUANCE:
Some claims deserve context beyond a simple true/false verdict. Use the \
"nuance" field to note important context that the verdict alone doesn't \
capture. Examples:
- Hyperbolic claims: "The specific number is hyperbolic. He is mentioned \
in the files, but hundreds of times — not a million."
- Technically true but misleading: "While technically accurate, this omits \
the key context that..."
- Wrong on specifics, right on substance: "The exact figure is wrong, but \
the underlying claim that spending was massive is supported."

Only include nuance when it adds genuine value. If the verdict speaks for \
itself, set nuance to null.

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
  "reasoning": "Brief explanation of how the evidence supports your verdict",
  "nuance": "Optional context note — hyperbole, missing context, etc. Set to null if not needed."
}

Return ONLY the JSON object. No markdown, no explanation, no wrapping.\
"""

JUDGE_USER = """\
Judge this sub-claim based ONLY on the evidence below. Do not use your own knowledge.

Original claim (for context): {claim_text}

Sub-claim to judge: {sub_claim}

Evidence:
{evidence_text}

Interpret the sub-claim in the context of the original claim. Think \
carefully about what the evidence says. Weigh conflicting sources. \
Then return a JSON object with "verdict", "confidence", "reasoning", \
and "nuance".\
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
# STEP 4: SYNTHESIZE (unified — works for both intermediate and final)
# =============================================================================

SYNTHESIZE_SYSTEM = """\
You are an impartial fact-checker. You have received verdicts for sub-claims \
and must combine them into a single verdict.

{synthesis_context}

CRITICAL — WEIGH BY IMPORTANCE, NOT BY COUNT:
Do NOT simply count true vs false sub-claims. Instead:

1. Identify the CORE ASSERTION — what is the person fundamentally claiming?
2. Identify SUPPORTING DETAILS — who, when, how much, attribution specifics.
3. The verdict follows the CORE ASSERTION, not the count.

A wrong supporting detail does NOT flip a true core assertion. A wrong \
core assertion is NOT saved by true supporting details.

Ask yourself: "Would a reasonable person say this claim is basically right \
or basically wrong?" That determines the verdict.

Example: "Fort Knox gold hasn't been audited despite promises by Trump \
and Elon Musk"
- Core assertion: gold hasn't been audited → TRUE ← this drives the verdict
- Supporting: Trump promised → TRUE
- Supporting: Musk promised → FALSE (Trump said Musk would, not Musk himself)
→ Verdict: "mostly_true" — the substance is correct. The Musk attribution \
is a minor inaccuracy that belongs in the nuance, not the verdict.

Another example: "NASA landed on Mars in 2019"
- Core: NASA landed on Mars → FALSE ← this drives the verdict
- Detail: year is 2019 → irrelevant since core is false
→ Verdict: "false" regardless of details.

Verdict scale:
- "true" — Core assertion AND key details are well-supported
- "mostly_true" — Core assertion is right, minor details wrong or imprecise
- "mixed" — Core assertion is genuinely split (not just detail errors)
- "mostly_false" — Core assertion is wrong, even if some details are right
- "false" — Core assertion AND details are clearly contradicted
- "unverifiable" — Not enough evidence to judge either way

The overall confidence should reflect the weakest link — if one sub-claim \
is very uncertain, your overall confidence should be lower.

NUANCE:
Sub-claims may include nuance notes (e.g., "this is hyperbolic but the \
underlying point is valid"). When synthesizing, weave these into an overall \
nuance note that gives the reader the REAL story. The nuance should feel \
like a knowledgeable friend explaining: "Look, the specific claim is wrong, \
but here's what's actually true..."

If any sub-claim has important nuance, you MUST include an overall nuance \
field. If no sub-claims have nuance and the verdict is straightforward, \
set nuance to null.

Confidence scoring (USE THE FULL RANGE):
- 0.95-1.0 — All sub-claims have rock-solid verdicts. Reserve for slam-dunks.
- 0.80-0.94 — Strong but not perfect. Most sub-claims well-supported.
- 0.60-0.79 — Moderate. Some sub-claims uncertain or evidence is mixed.
- 0.40-0.59 — Weak. Significant uncertainty in multiple sub-claims.
- Below 0.40 — Very uncertain. Mostly guesswork.

Do NOT default to 0.9+. Be honest about uncertainty.

Return a JSON object:
{{
  "verdict": "true|mostly_true|mixed|mostly_false|false|unverifiable",
  "confidence": 0.0 to 1.0,
  "reasoning": "Brief summary of how the sub-verdicts combine",
  "nuance": "Overall context note synthesizing sub-claim nuances. Null if not needed."
}}

Return ONLY the JSON object. No markdown, no explanation, no wrapping.\
"""

SYNTHESIZE_USER = """\
Combine these sub-claim verdicts into a single verdict.

{synthesis_framing}

Sub-claim verdicts:
{sub_verdicts_text}

Return a JSON object with "verdict", "confidence", "reasoning", and "nuance".

/no_think\
"""

# Why a unified synthesis prompt?
#   The same operation happens at every level of the tree: "here are child
#   verdicts, combine them." The only difference is context framing:
#   - Final: "This is the overall verdict for the claim."
#   - Intermediate: "This is a verdict for one aspect. It will be combined
#     with other aspects later."
#
#   The activity formats {synthesis_context} and {synthesis_framing} based
#   on whether it's a final or intermediate synthesis. The core reasoning
#   (importance weighting, confidence scoring, nuance) is identical.
#
# Why the full 6-level scale at every level?
#   The old approach used 4 levels (true/false/partially_true/unverifiable)
#   for intermediate nodes and 6 for final. This lost expressiveness — an
#   intermediate node couldn't distinguish "mostly true with minor issues"
#   from "genuinely mixed." Now every level uses the same scale, so nuance
#   is preserved as it flows up the tree.
