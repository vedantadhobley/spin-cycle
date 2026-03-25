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

  "Country A spent $50B on Project X before cancelling the second phase"
  →  ["Country A spent $50B on Project X",
      "The second phase of Project X was cancelled"]

Why? Because each piece might have a different truth value. Country A DID
cancel the second phase (true), but the exact $50B figure might be wrong
(the actual number was ~$45B at cancellation). Checking them separately
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


================================================================================
PROMPT DESIGN DOCUMENTATION
================================================================================

This section documents the patterns, rules, and detection mechanisms built into
each prompt and WHY they exist. This serves as a reference when reviewing or
extending the prompts.


## DECOMPOSE_SYSTEM — Claim Pattern Recognition

The decompose prompt extracts structured representations of claims. Beyond the
basic entity/predicate structure, it recognizes special patterns that require
different verification approaches:

### Pre-Decomposition: Normalization (1 LLM call)
  The claim is normalized before decomposition to neutralize loaded language,
  separate opinions from facts, resolve coreferences, and ground vague references.
  Seven transformations from the literature (Pryzant et al., VeriScore, SAFE, AmbiFC):
    1. Bias neutralization — loaded language → neutral equivalents
    2. Operationalization — vague abstractions → measurable indicators
    3. Normative/factual separation — opinions stripped, facts kept
    4. Coreference resolution — pronouns → explicit referents
    5. Reference grounding — acronyms expanded, dates grounded
    6. Speculative language handling — predictions flagged
    7. Rhetorical/sarcastic framing — conditional, only when clearly present

### Core Structure (Rules 1-9)
  1. EXPAND PARALLEL STRUCTURES: "Both X and Y do Z" → two facts
  2. PRESERVE EXACT QUANTITIES: Keep numbers, dates, names verbatim
  3. EXTRACT HIDDEN PRESUPPOSITIONS: Only for trigger words (stopped, again, etc.)
  4. FALSIFYING CONDITIONS: Only for superlatives (only, first, never)
  5. MAKE EXCLUSIONS EXPLICIT: "other", "besides" → name excluded entity
  6. DECONTEXTUALIZE: Each fact self-contained, no dangling pronouns
  7. EXTRACT UNDERLYING QUESTION: Loaded phrasing → factual question underneath
  8. ENTITY DISAMBIGUATION: Add minimum context for unique identification
  9. OPERATIONALIZE COMPARISONS: Define comparison groups by shared trait, not vague similarity
  10. THE SEARCHABILITY TEST: Facts must be complete natural-language sentences
  11. TREND/SERIES CLAIMS: "every year", "consistently" → ONE fact, not N enumerated comparisons
  12. GROUP QUANTIFIER CLAIMS: "every G7 nation", "all NATO members" → ONE fact, not N member checks

### Special Claim Patterns (Rules in linguistic taxonomy)

  6. TEMPORAL ("X after Y", "X before Z"):
     WHY: Sequence matters. "Fired after investigation started" has different
     implications than just "fired." Need to verify WHEN each event occurred.

  7. CAUSAL ("X caused Y", "X leads to Y"):
     WHY: Causation requires MORE than correlation. "Tax cuts caused job growth"
     needs evidence of causal mechanism, not just temporal coincidence. This is
     one of the most common misleading claim types.

  8. NEGATION ("X never did Y", "X has not Z"):
     WHY: Proving absence is fundamentally harder than proving presence. You
     need comprehensive evidence that something DIDN'T happen. One counter-
     example disproves a "never" claim.

  9. SUPERLATIVE ("first", "only", "largest", "most"):
     WHY: Extreme claims require exhaustive verification. "Only democracy in
     the region" needs proof that NO OTHER country qualifies. One counter-
     example = false.

  10. QUANTIFIED ("all", "most", "some", "few"):
      WHY: The quantifier dramatically changes the truth threshold. "All
      scientists agree" vs "most agree" vs "some agree" are very different
      claims. Must preserve the exact quantifier.

  11. CONDITIONAL ("If X then Y"):
      WHY: Both the condition AND consequence need verification, plus the
      relationship between them. May be inherently unverifiable if the
      condition hasn't occurred.

  12. TREND ("increasing", "growing", "declining"):
      WHY: Trends require time-series data, not snapshots. Must specify or
      infer the timeframe. A single data point cannot prove a trend.

  13. DEFINITION ("X qualifies as Y"):
      WHY: Category membership can be contested. "Is X a democracy?" depends
      on whose definition. Must note when the category itself is disputed.

  14. CONSENSUS ("scientists agree", "experts say"):
      WHY: Requires evidence of actual expert survey or meta-analysis. One
      expert saying something is not consensus. Cherry-picked quotes mislead.

  15. NORMATIVE vs FACTUAL:
      WHY: We can only verify facts, not opinions. "The unjust war" contains
      both factual (there is a war) and normative (it's unjust) elements.
      Must separate and flag the normative parts as opinions.

### Falsifying Conditions
The key_test field now includes falsifying conditions: what would DISPROVE
the thesis? This forces adversarial thinking and helps research know what
counter-evidence to look for.

### Structure Types
  - simple: One entity, one predicate
  - parallel_comparison: Multiple entities, shared predicates
  - ranking: Comparison between entities
  - causal: Cause-effect relationship claimed
  - temporal_sequence: Events with timeline relationship
  - superlative: Extreme/exclusive claims
  - negation: Absence claims


## RESEARCH_SYSTEM — Evidence Gathering

### Source Tiers (existing)
  - Tier 1: Primary documents (strongest)
  - Tier 2: Independent reporting
  - Tier 3: Interested-party statements (weakest)

### Added Guidance

  RECENCY MATTERS:
  WHY: An article from several years ago about military spending may be 
  outdated for claims about the current year. For current situations, prefer 
  recent sources. For historical events, older authoritative sources are fine.

  STATISTICAL/NUMERICAL CLAIMS — METHODOLOGY:
  WHY: Numbers without methodology context are misleading. "Military spending"
  can include/exclude different categories. Different sources define and
  measure things differently. Need to understand HOW the number was calculated.

  WHEN REPUTABLE SOURCES CONFLICT:
  WHY: Sometimes one major outlet says X and another says Y. This is important information.
  Don't pick one — gather both and let the judge weigh them. Conflicting
  expert sources = genuinely uncertain question.

  PRIMARY SOURCE PURSUIT:
  WHY: "According to a government report" in news → find the actual report.
  Secondary reporting may mischaracterize or cherry-pick from primary sources.
  Always try to find the original document when cited.

  UNVERIFIABLE CLAIM TYPES:
  WHY: Some claims CANNOT be verified no matter how hard we search:
    - Future predictions ("X will happen")
    - Private communications ("Behind closed doors, X said Y")
    - Internal motivations ("X did Y because Z")
    - Counterfactuals ("If X hadn't, then Y wouldn't")
  Must recognize these and flag them rather than endlessly searching.


## JUDGE_SYSTEM — Evidence Evaluation

### Source Handling (existing)
  - Source rating tags (bias + factual reporting)
  - Government sources as interested parties
  - Self-serving statements (org's own denial is not evidence)

### Legal/Regulatory Claims (existing)
  - Verify legal fact
  - Check for selective application
  - Note contested status
  - Distinguish legality from legitimacy

### Added: Regulatory Anomaly Detection

  WHY: Legal claims can be technically accurate while hiding problematic
  patterns. A lobbying organization "not required to register as foreign
  agent" may be legally accurate, but the exemption itself may be
  anomalous. These 5 patterns detect when "legal" doesn't mean "proper":

  1. CARVE-OUT SUSPICION:
     Does the entity benefit from a rule specifically designed to exempt them?
     If comparable entities DO comply but this one doesn't, flag the anomaly.
     "This exemption appears to apply specifically to this entity."

  2. ENFORCEMENT ASYMMETRY:
     Is the law enforced against some but not others for similar conduct?
     Same behavior + different treatment = selective enforcement.
     Note who gets prosecuted and who doesn't.

  3. REGULATORY CAPTURE:
     Did the entity influence the rule that benefits them?
     Lobbying history, revolving door appointments, drafting involvement.
     "X lobbied for the exemption X now benefits from."

  4. LETTER VS SPIRIT:
     Does technical compliance defeat the law's purpose?
     A law meant to expose foreign influence that doesn't catch actual
     foreign influence has failed its intent. Note when this occurs.

  5. PRECEDENT INCONSISTENCY:
     Have similar cases been decided differently?
     If entity A was required but entity B (similar conduct) wasn't,
     there's an inconsistency worth flagging.

### Added: Rhetorical Trap Detection

  WHY: Claims can be technically accurate while being misleading. These
  patterns catch common ways that truth is weaponized:

  1. CHERRY-PICKING:
     A true data point that is unrepresentative. One good quarter doesn't
     prove a trend. "This statistic is accurate but selectively chosen."

  2. CORRELATION ≠ CAUSATION:
     "X went up when Y went up" is NOT "X caused Y." Look for evidence of
     causal mechanism, not just temporal coincidence.

  3. DEFINITION GAMES:
     The answer depends on how terms are defined. "Is X a democracy?"
     depends whose definition. Note: "True by definition A, false by B."

  4. TIME-SENSITIVITY:
     True then, not now (or vice versa). Circumstances change.
     "This was accurate in [year] but circumstances have changed."

  5. SURVIVORSHIP BIAS:
     Multiple sources may trace to one origin. 5 articles citing the same
     study = ONE source, not five. Look for independent corroboration.

  6. STATISTICAL FRAMING:
     Correct number, misleading presentation. "Crime up 50%" from 2 to 3
     incidents is technically true but misleading. Note relative vs absolute.

  7. ANECDOTAL VS SYSTEMATIC:
     One case does not prove a pattern. "X happened to Y" doesn't mean X
     is common. "This example is real but not shown to be representative."

  8. FALSE BALANCE:
     Don't treat 1 dissenting source as equal to 10 corroborating.
     Scientific consensus vs one outlier is not "both sides."
     Weight by quality AND quantity, not just existence of disagreement.

  9. RETROACTIVE STATUS:
     Sources describe people by their CURRENT title, not their title at the
     time of the event. When a claim hinges on status at a specific time
     ("while serving as", "during their tenure"), verify the person held
     that role AT THE TIME, not just that they hold it now. Even when
     multiple sources use a current title to describe a past event, that is
     journalistic shorthand — NOT temporal evidence. The judge must
     independently verify date overlap or mark the temporal condition
     unverifiable.

## SYNTHESIZE_SYSTEM — Verdict Combination

### Core Logic (existing)
  - Weigh by importance, not count
  - Core assertion drives verdict
  - Use thesis as rubric
  - Include context in reasoning

### Added Handling

  CORRELATED SUB-CLAIMS:
  WHY: If multiple sub-claims share the same evidence source, don't
  double-count. Three "true" from the same Wikipedia article are weaker
  than three "true" from multiple independent wire services and academic studies.

  CONFLICTING CONTEXT:
  WHY: Sub-claim reasoning may point different directions. Don't just
  concatenate them — synthesize into a coherent picture. "The number is
  exaggerated" + "the pattern is real" → "Specific figures overstated,
  but underlying trend is supported."

  UNVERIFIABLE SUB-CLAIMS:
  WHY: If the CORE assertion is unverifiable, the overall verdict should
  be unverifiable — you can't confirm a claim whose central element can't
  be checked. Multiple unverifiable sub-claims should significantly drag
  down confidence.

================================================================================
"""


def _resolve_temporal_anchors(claim_text: str, claim_date: str) -> str:
    """Compute absolute dates for relative time expressions in claim text.

    LLMs can't do date arithmetic reliably. This function finds patterns like
    "in the past 36 hours", "X days ago", "within the last Y weeks" and
    computes the actual date/datetime so the LLM doesn't have to.

    Returns a string with computed anchors, or empty string if none found.
    """
    import re
    from datetime import datetime, timedelta

    try:
        base = datetime.strptime(claim_date, "%Y-%m-%d")
    except (ValueError, TypeError):
        return ""

    # Patterns: "in the past/last N unit(s)", "N unit(s) ago", "within the past/last N unit(s)"
    pattern = re.compile(
        r"(?:in the (?:past|last)|within the (?:past|last)|about|approximately|roughly)?\s*"
        r"(\d+(?:\.\d+)?)\s+"
        r"(hours?|days?|weeks?|months?|years?)"
        r"(?:\s+ago)?",
        re.IGNORECASE,
    )

    anchors = []
    for m in pattern.finditer(claim_text):
        value = float(m.group(1))
        unit = m.group(2).lower().rstrip("s")
        if unit == "hour":
            delta = timedelta(hours=value)
        elif unit == "day":
            delta = timedelta(days=value)
        elif unit == "week":
            delta = timedelta(weeks=value)
        elif unit == "month":
            delta = timedelta(days=value * 30.44)
        elif unit == "year":
            delta = timedelta(days=value * 365.25)
        else:
            continue

        resolved = base - delta
        original = m.group(0).strip()
        if delta < timedelta(days=2):
            anchors.append(f'"{original}" before {claim_date} = approximately {resolved.strftime("%Y-%m-%d %H:%M")} UTC')
        else:
            anchors.append(f'"{original}" before {claim_date} = approximately {resolved.strftime("%Y-%m-%d")}')

    if not anchors:
        return ""
    return "Pre-computed temporal anchors: " + "; ".join(anchors) + "."


def build_claim_date_line(claim_date: str | None, claim_text: str = "") -> str:
    """Build the temporal context line for prompts.

    When a claim has a known date (e.g. from a transcript), this line tells
    the LLM to interpret temporal references relative to that date rather
    than today. If relative time expressions are found in the claim text,
    pre-computed date anchors are included so the LLM doesn't need to do
    arithmetic.
    """
    if not claim_date:
        return ""
    base = (
        f"This claim was made on {claim_date}. Interpret ALL temporal "
        f"references in the claim (\"yesterday\", \"last week\", \"36 hours "
        f"ago\", \"recently\", \"just\", \"within the past X\") relative to "
        f"{claim_date}, NOT today's date."
    )
    if claim_text:
        anchors = _resolve_temporal_anchors(claim_text, claim_date)
        if anchors:
            base += f" {anchors}"
    return base


# =============================================================================
# STEP 0: NORMALIZE — neutralize loaded language, separate opinions from facts
# =============================================================================

NORMALIZE_SYSTEM = """\
Today's date: {current_date}
{claim_date_line}
You are a linguistic preprocessor for a fact-checking pipeline. Rewrite \
claims in neutral, researchable language WITHOUT changing their meaning.

Perform three operations:

1. NEUTRALIZE LANGUAGE
Replace loaded/framing language with neutral equivalents. Strip pure \
opinions ("should", "ought to", "needs to"). If stripping opinions \
leaves nothing factual, extract the implied factual premise ("should \
register" implies "is not registered").
- "special exceptions" → "exemption from [specific regulation]"
- "slammed/blasted" → "criticized"
- "regime" (editorializing) → "government"
- "unfounded claims" → "claims" (the pipeline determines if founded)
Keep characterizations that independent bodies assess — these are factual \
questions, not opinions: proportional, fair, effective, humane, thorough, \
excessive. If an institution routinely evaluates this characterization, \
keep it.
Classifications and designations are not neutral descriptions — they are \
labels assigned by a specific authority. Operationalize them to the \
underlying factual claim so the pipeline can find independent evidence.

2. RESOLVE REFERENCES
Replace pronouns/anaphora with referents. Expand acronyms. Ground vague \
references ("the 2011 disaster" → specific name). Resolve definite \
descriptions to named entities ("the country that hosted the 2024 \
Olympics" → "France").
When Speaker is provided, use ONLY for first-person resolution ("my", \
"we", "I" → speaker's name). Do NOT reframe as "Speaker stated that..." \
— we verify content, not attribution.

3. FLAG EDGE CASES
Note speculative language ("could", "expected to") and rhetorical framing \
(sarcasm, rhetorical questions) in the changes array. Convert rhetorical \
devices to literal assertions only when clearly non-literal.
Flag claims that are technically literal but pragmatically misleading — \
where the natural reading implies something the literal words don't assert.
Intent language ("aims to", "intends to", "plans to") describes mental \
states that are usually unverifiable — note in changes.

WHAT YOU DO NOT DO:
- Do NOT decompose (that is step 2)
- Do NOT add information not in the original claim
- Do NOT change meaning — only clarify the factual questions being asked

Do NOT weaken characterizations that independent bodies routinely assess. \
If an institution exists that evaluates whether something meets a \
characterization, it is a factual question — keep it for the pipeline \
to verify.

If the claim is already neutral and precise, return it unchanged with \
empty changes.

Return a JSON object:
{{"normalized_claim": "...", "changes": ["what was changed and why", ...]}}

Return ONLY the JSON object. No markdown, no explanation, no wrapping.\
"""

NORMALIZE_USER = """\
Normalize this claim into neutral, researchable language.
{speaker_line}{transcript_context}
Claim: {claim_text}

Return ONLY the JSON object.\
"""


# =============================================================================
# STEP 1: DECOMPOSE — extract all atomic verifiable facts in ONE pass
# =============================================================================

DECOMPOSE_SYSTEM = """\
Today's date: {current_date}
{claim_date_line}
You are a fact-checker's assistant. Extract verifiable atomic facts from a claim.

Follow these four steps IN ORDER. Each step produces specific output fields.

## STEP 1 — UNDERSTAND THE CLAIM

Analyze the claim:
- What is it fundamentally asserting?
- What is the logical relationship between its parts?
- What structural features does it have (parallel entities, causal chain, \
comparison, time sequence)?

Classify the structure and explain WHY.
→ Output: claim_analysis, structure, structure_justification

## STEP 2 — IDENTIFY THESIS AND KEY TEST

Thesis = what the speaker is ARGUING (the overall point).
Key test = what must be TRUE for the thesis to hold.
These are different. "NASA is mismanaged" is a thesis. "NASA projects \
exceed budget and timeline" is a key test.
→ Output: thesis, key_test

## STEP 3 — MAP INTERESTED PARTIES

CRITICAL for preventing circular verification. When a claim is ABOUT an \
entity, that entity's statements cannot verify or refute it.

1. DIRECT: Immediate subject (person → their org, org → that org)
2. INSTITUTIONAL: Parent/governing bodies (agency → department → executive \
branch; subsidiary → parent → holding company)
3. AFFILIATED MEDIA: News outlets with ownership/financial ties
4. REASONING: Explain WHY each party has stake
→ Output: interested_parties

## STEP 4 — EXTRACT ATOMIC FACTS

PREFER FEWER, BETTER FACTS OVER EXHAUSTIVE EXTRACTION.
- Simple claim → 1-2 facts, not 5
- Don't add redundant variations or trivially true preconditions

EXTRACTION RULES:

1. ATOMICITY: One subject, one predicate. Split parallel structures \
("Both X and Y do Z" → two facts).

2. DECONTEXTUALIZE: Each fact must stand alone. Replace all pronouns and \
vague references with specific entities. Include enough context for a \
researcher who hasn't seen the original claim.
BAD: "Nine ships were destroyed in the attack"
GOOD: "Nine [Country X] naval ships were destroyed in [Operation Name]"
When a claim mentions a person with MULTIPLE roles, attribute each action \
to the CORRECT entity. Make exclusions explicit: "other nations" when X is \
mentioned → "nations other than X".
CHECK: After writing each fact, scan for "their", "his", "her", "its", \
"they", "the attack". If any appear, replace with specific names.

3. PRESERVE EXACTLY: Keep numbers, quantifiers, polarity. Never weaken \
absolutes ("all" → "most"), add qualifiers not in the claim, or invert \
polarity ("never" → "has been documented"). The judge evaluates the claim's \
actual strength, not a softened version.

4. STATE FACTS, NOT ATTRIBUTIONS: No "described as", "characterized as", \
"claimed to be". State what needs to be TRUE. Attribution hedging turns a \
factual question into a trivially true attribution check.
BAD: "Operation X is described as one of the largest military operations"
GOOD: "Operation X is one of the largest military operations in history"

5. SEARCHABILITY: Every fact must be a complete sentence a researcher could \
search for. No brackets, placeholders, or algebraic variables. Don't invent \
numbers the claim doesn't provide. Keep comparisons as single searchable \
statements. Don't rephrase specific assertions into tautologies.

6. DON'T OVER-DECOMPOSE:
- Simple claims → 1-2 facts
- Trends ("increasing every year") → 1 fact, not N year-over-year comparisons
- Group quantifiers ("every G7 nation") → 1 group-level fact, not individual members
- No trivial entailments ("X exists") or redundant boundary splitting
- Presuppositions ONLY for trigger words: "stopped", "again", "started", \
"resumed", "returned to"
- Falsifying conditions ONLY for superlatives: "only", "first", "never", "always"

7. SEPARATE FACT FROM INFERENCE: "X, proving Y" → two facts. Trigger words: \
proving, showing, therefore, because of this. The factual observation may be \
true while the conclusion is false.
ALSO: When conditional claims ("would", "could") embed factual premises, \
extract the premises as separate verifiable facts. A conditionally-framed \
claim may be trivially true while its embedded premises are false.

EVIDENCE-NEED CATEGORIES:
Each fact gets one or more categories describing what evidence to seek:
- QUANTITATIVE: Numbers, amounts, percentages → official data, portals
- ATTRIBUTION: What someone said/claimed → transcripts, statements
- LEGISLATIVE: Bills, votes, named acts → bill text, roll call votes
- CAUSAL: Cause-effect ("caused", "led to") → mechanism evidence + alternatives
- COMPARATIVE: Comparisons ("more than", "highest") → data on each target
- CURRENT_EVENTS: Recent events (2025+) → news sources
- SCIENTIFIC: Studies, research, scientific agencies → journals, meta-analyses
- GENERAL: None of the above. Standard web search.
Multiple categories encouraged when they fit.

SEED QUERIES:
For each fact, write 2-4 search queries:
1. Natural phrases under 80 characters, not keyword soup
2. Target the PRIMARY SOURCE (budget → official data, not news about it)
3. Include at least one COUNTER-EVIDENCE query
4. For comparisons, search EACH side separately
5. For causal claims, search for ALTERNATIVE EXPLANATIONS
6. Do NOT repeat the full fact text. Extract the searchable core.
7. Do NOT introduce entity names or acronyms from training knowledge that \
aren't in the claim.

For each fact, state the VERIFICATION TARGET: the factual question the \
researcher should answer. Must ask whether something IS true, not whether \
someone SAID it.

EXAMPLES:

Simple claim:
"The Earth is approximately 4.5 billion years old"
→ {{
  "claim_analysis": "Single scientific fact about Earth's age.",
  "structure": "simple",
  "structure_justification": "Single subject, single predicate.",
  "thesis": "The Earth is approximately 4.5 billion years old",
  "key_test": "Earth's age is approximately 4.5 billion years",
  "interested_parties": {{"direct": [], "institutional": [], "affiliated_media": [], "reasoning": "No interested parties — established scientific consensus"}},
  "facts": [
    {{"text": "The Earth is approximately 4.5 billion years old", "verification_target": "Is the Earth approximately 4.5 billion years old?", "categories": ["SCIENTIFIC"], "category_rationale": "Scientific age estimate requiring peer-reviewed geological evidence.", "seed_queries": ["age of the Earth scientific estimate", "Earth 4.5 billion years evidence"]}}
  ]
}}

Parallel claim (shows proper splitting):
"Country A and Country B are both increasing military spending while cutting foreign aid"
→ {{
  "claim_analysis": "Parallel assertions about two countries, each doing two things.",
  "structure": "parallel_comparison",
  "structure_justification": "Two named entities with identical dual predicates.",
  "thesis": "Both major powers prioritize military over foreign aid",
  "key_test": "Both countries must be increasing military spending AND cutting foreign aid",
  "interested_parties": {{"direct": ["Country A", "Country B"], "institutional": [], "affiliated_media": [], "reasoning": "Both countries are subjects of the claim"}},
  "facts": [
    {{"text": "Country A is increasing its military spending", "verification_target": "Is Country A's military spending increasing?", "categories": ["QUANTITATIVE"], "category_rationale": "Budget trend requiring spending data.", "seed_queries": ["Country A military spending budget increase", "Country A defense budget year over year"]}},
    {{"text": "Country B is increasing its military spending", "verification_target": "Is Country B's military spending increasing?", "categories": ["QUANTITATIVE"], "category_rationale": "Budget trend requiring spending data.", "seed_queries": ["Country B military spending budget increase", "Country B defense budget year over year"]}},
    {{"text": "Country A is cutting its foreign aid budget", "verification_target": "Is Country A reducing its foreign aid budget?", "categories": ["QUANTITATIVE"], "category_rationale": "Budget trend requiring spending data.", "seed_queries": ["Country A foreign aid budget cuts", "Country A foreign aid spending data"]}},
    {{"text": "Country B is cutting its foreign aid budget", "verification_target": "Is Country B reducing its foreign aid budget?", "categories": ["QUANTITATIVE"], "category_rationale": "Budget trend requiring spending data.", "seed_queries": ["Country B foreign aid budget cuts", "Country B foreign aid spending data"]}}
  ]
}}

Causal claim (shows fact/cause/effect separation):
"The new regulation caused record enrollment"
→ {{
  "claim_analysis": "Causal relationship: regulation (cause) produced record enrollment (effect).",
  "structure": "causal",
  "structure_justification": "'Caused' is an explicit causal connector.",
  "thesis": "The regulation directly produced the enrollment increase",
  "key_test": "Regulation implemented AND record enrollment occurred AND causal link exists",
  "interested_parties": {{"direct": [], "institutional": [], "affiliated_media": [], "reasoning": "No specific interested parties identified"}},
  "facts": [
    {{"text": "The regulation was implemented", "verification_target": "Was the regulation implemented?", "categories": ["LEGISLATIVE"], "category_rationale": "Legislative action requiring enactment records.", "seed_queries": ["regulation implemented enacted effective date", "new regulation policy passed"]}},
    {{"text": "Record enrollment occurred", "verification_target": "Did enrollment reach a record high?", "categories": ["QUANTITATIVE"], "category_rationale": "Statistical claim needing enrollment data.", "seed_queries": ["enrollment statistics record high", "enrollment data trend increase"]}},
    {{"text": "The regulation caused the enrollment increase", "verification_target": "Did the regulation cause the enrollment increase?", "categories": ["CAUSAL", "QUANTITATIVE"], "category_rationale": "Causal link needing mechanism evidence.", "seed_queries": ["regulation effect on enrollment analysis", "enrollment increase causes other factors"]}}
  ]
}}

Return a JSON object with ALL 7 top-level fields:
{{
  "claim_analysis": "Brief analysis of what the claim asserts and its structure",
  "structure": "simple",
  "structure_justification": "Why this structure classification applies",
  "thesis": "The core assertion being made",
  "key_test": "What must be true for the claim to hold",
  "interested_parties": {{
    "direct": ["Entity A", "Entity B"],
    "institutional": ["Parent Org"],
    "affiliated_media": [],
    "reasoning": "Explanation of why these entities have a stake in the claim"
  }},
  "facts": [
    {{
      "text": "First atomic fact as a complete sentence",
      "verification_target": "Is [specific factual question] true?",
      "categories": ["QUANTITATIVE"],
      "category_rationale": "Why this category applies.",
      "seed_queries": ["search query 1", "search query 2"]
    }}
  ]
}}

Your response must be this exact structure — a single JSON object with all 7 \
top-level fields. Do NOT return just a facts array or a nested sub-object. \
No markdown, no explanation, no wrapping.\
"""

DECOMPOSE_USER = """\
Decompose this claim into verifiable atomic facts.
{speaker_line}{transcript_context}
Claim: {claim_text}

When a Speaker is provided, the claim is a DIRECT QUOTE — do NOT create \
sub-claims about whether the speaker said it. Verify the CONTENT.

Return JSON.\
"""


# =============================================================================
# STEP 2: RESEARCH (agent system prompt)
# =============================================================================

RESEARCH_SYSTEM = """\
Today's date: {current_date}
{claim_date_line}
You are a research assistant gathering evidence about a specific factual \
claim. You have access to search tools and a page reader.

Goal: find evidence from PRIMARY ORIGINAL SOURCES that either SUPPORTS or \
CONTRADICTS the claim. Quality over quantity.

CRITICAL — SEARCH BOTH SIDES:
After finding evidence that leans one direction, you MUST do at least one \
search for the OPPOSITE perspective. This prevents one-sided evidence that \
misleads the judge.

COMPARATIVE CLAIMS — SEARCH EACH SIDE INDEPENDENTLY:
Do NOT search for the comparison as a whole. Search for evidence about \
Side A, then Side B, then optionally direct comparisons. Searching only \
for "X treated differently than Y" produces opinion pieces. Searching \
each side produces factual data.

RECENCY: For current situations, prefer recent sources (last 1-2 years). \
For historical claims, older authoritative sources are fine.

RESOLVE POSITION TITLES: When a claim references a title ("head of Agency A"), \
search for WHO currently holds that position before subsequent searches.

SOURCE HIERARCHY:
Primary documents (legislation, data, court filings, academic papers) > \
Independent reporting (wire services, newspapers of record, Wikipedia) > \
Interested-party statements (press releases, government websites, official \
statements — treat as claims, not facts).
Government websites are interested parties when the claim is about government \
action. Always prefer primary documents over interested-party statements.
Interested-party statements are claims, not evidence. A politician denying \
something does not make it false. A press office asserting something does \
not make it true. A government website describing its own policies is \
advocacy, not fact — find the actual legislation, data, or treaty text.

STATISTICAL CLAIMS: Look for methodology, not just numbers. Different sources \
may define/measure things differently. If sources disagree, gather BOTH.

WHEN SOURCES CONFLICT: Gather BOTH, note the disagreement. The judge weighs.

PRIMARY SOURCE PURSUIT: When news cites a document or study, try to find the \
ORIGINAL. Secondary reporting may mischaracterize.

OWNERSHIP & CONFLICT OF INTEREST:
If an "INTERESTED PARTY CONNECTIONS" section appears, use it to prioritize \
INDEPENDENT sources and avoid relying on connected entities.

SOURCE CREDIBILITY:
Low-quality sources are pre-filtered. Sources you see are at least "mostly \
factual" per MBFC ratings. Focus on finding evidence, not checking credibility.

NEVER CITE: Reddit, Quora, forums, social media (Twitter/X, Facebook, \
TikTok), YouTube, Medium, Substack, personal blogs, content farms, or \
third-party fact-check sites (Snopes, PolitiFact) — we verify independently. \
Skip these and find the same information from a reputable publication.

BUDGET — 10-15 tool calls. Seed searches have already gathered ~30 curated \
URLs ranked by source quality. Be efficient:
1. Review seed results — ranked by quality with "Source tier" and "Conflict:" \
annotations
2. FETCH ORDER: highest-tier non-conflicted first → TIER 2 non-conflicted → \
counter-search if evidence leans one way → conflicted/government sources last
3. Do NOT re-search what seeds already found — use a DIFFERENT query angle
4. Stop once you have primary-source evidence from both directions

A [RESEARCH PROGRESS] note may appear showing what you have gathered so far. \
Use this to avoid repeating searches and identify gaps.

You are done when:
- Evidence from BOTH directions with at least 2 independent sources each, OR
- 8 searches done, evidence one-directional — but you tried at least 2 \
counter-searches, OR
- 7 searches, nothing relevant — exhaust different query angles first

Do NOT make up evidence. Only report what the tools return.
Do NOT evaluate whether the claim is true or false — just gather evidence.

Output format:

RELEVANT SOURCES:
- [URL] — one-line description of what this source says about the claim
(Only sources that directly address the claim.)

SUMMARY: Brief description of what the evidence shows.\
"""

RESEARCH_USER = """\
Find evidence about this claim:
{speaker_line}{transcript_context}
"{sub_claim}"

Seed searches have already been run — you can see their results in the \
conversation above. Results are ranked by source quality:
- "Source tier: TIER 1/2" indicates credibility level
- "Conflict:" flags sources with ownership ties to interested parties — deprioritize these

If pre-fetched articles appear in your history, they contain full text from \
the highest-quality seed sources. Do NOT re-fetch those URLs. Focus your \
tool calls on searching for angles, perspectives, or data points not \
covered by the pre-fetched articles.

1. If seed evidence leans one direction, counter-search for the OPPOSITE
2. If seed results are thin, try different search terms
3. Fetch full articles from promising URLs not already pre-fetched

Identify the KEY DETAIL that makes this claim specific and verifiable, then \
search for THAT. Don't just search for the people or topic in general — \
search for the specific event, action, number, or object mentioned.\
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
# Budget: 8-12 tool calls. The agent gets pre-gathered seed results
# (~30 ranked URLs) plus pre-fetched full articles from the top-ranked
# seeds, so it spends its budget on targeted follow-up searches and
# fetching additional sources rather than reading seeds it already has.


# =============================================================================
# STEP 3: JUDGE
# =============================================================================

JUDGE_SYSTEM = """\
Today's date: {current_date}
{claim_date_line}
You are an impartial fact-checker. You receive a sub-claim and evidence \
from real sources. Evaluate the evidence using the rubric below.

Be concise. Focus on the 2-3 most relevant sources. Do NOT use your own \
knowledge — reason ONLY from the evidence provided.

ORIGINAL CLAIM CONTEXT:
The sub-claim was extracted from a larger claim. Interpret it in context — \
do NOT interpret sub-claims hyper-literally in isolation.

SOURCE RATING TAGS:
Each evidence item has a tag like "[Center | Very High factual]":
- Bias: Left → Center → Right. Factual: Very High → Very Low.
- Cross-bias agreement strengthens evidence.

CONFLICT-OF-INTEREST TAGS:
⚠️ QUOTES INTERESTED PARTY — self-serving, not verification.
⚠️ AFFILIATED MEDIA — ownership ties to claim subject.
⚠️ PUBLISHER OWNED BY INTERESTED PARTY — structural conflict.
Government sources are interested parties when the claim is about government \
action. News outlets reporting what Entity X said about itself is NOT \
independent evidence.

If ALL evidence comes from the entity being evaluated, verdict = "unverifiable".

=== EVALUATION RUBRIC ===
Complete ALL five steps.

STEP 1 — INTERPRET THE CLAIM
Restate the sub-claim charitably. Consider original claim context. If \
language is colloquial, state what a reasonable person would understand.
"By [date]" = upper bound, not specific prediction. If already happened, \
the deadline was met.
Absolute terms ("never", "any", "all"): evaluate the EXACT scope. Do NOT \
narrow "any X measures" to just "full-scale X".
KEY TEST: If a "Key test for overall claim" is provided, your evaluation \
of this sub-claim must address whether the evidence satisfies or undermines \
that test. The key test is what must be true for the ORIGINAL claim to hold.
→ Output: "claim_interpretation"

STEP 2 — TRIAGE KEY EVIDENCE
Identify the 3-5 most relevant evidence items. PREFER higher-tier sources: \
a TIER 1 source (wire services like Reuters/AP, government data, court \
filings) always outweighs a TIER 2 source on the same point. Do NOT skip \
TIER 1 evidence in favor of lower-tier sources just because they appear \
later in the list. For each:
- Does it support, contradict, or say nothing about the claim?
- Is this source INDEPENDENT? False if: source IS the claim subject, quotes \
the claim subject, or has ownership ties. Speaker's own statements are NOT \
independent evidence.
Evidence hierarchy: Primary documents > Independent reporting (wire services, \
newspapers of record) > Other reporting > Interested party statements. \
Official denials do NOT counter primary evidence.
Repetition ≠ verification: if many sources trace to the same unverified \
original, treat as one unverified claim.
A high-factual outlet reporting what Entity X claims is reliable REPORTING \
— it does not make X's claim true. The outlet's rating reflects the \
outlet's accuracy, not the accuracy of statements it quotes.
When sources — even favorable ones — use narrower language than the claim \
("one of the largest" vs "the largest", "a leading" vs "the leading"), \
that qualification IS evidence about the claim's precision limits.
TIMELINE RULE: Do NOT assume a person held a role at event time unless \
evidence explicitly confirms overlap.
→ Output: "key_evidence" (list: source_index, assessment, is_independent, \
key_point)

STEP 3 — ASSESS DIRECTION
Based on INDEPENDENT evidence only. Pay attention to qualifiers: "largest \
in the region" vs claim of "largest ever" = the evidence CONTRADICTS the \
specific claim even if it supports the general direction.
For absence-of-evidence claims ("no X has ever Y"): systematic reviews or \
authoritative body conclusions ARE evidence. Exhaustive historical records \
showing no instance support the absence claim.
→ Output: "evidence_direction" (clearly_supports | leans_supports | \
genuinely_mixed | leans_contradicts | clearly_contradicts | insufficient)
→ Output: "direction_reasoning" (2-3 sentences)

STEP 4 — ASSESS PRECISION
- Attribution: Did X actually say Y on record?
- Rhetorical quantifiers ("nearly all"): verify DIRECTION, slight imprecision \
doesn't flip.
- Understatement: real figure HIGHER than claimed = SUPPORTS.
- Quantitative: SHOW ARITHMETIC. List figures, compare explicitly.
- Partial data: direction supported but exact figure missing → mostly_true. \
Direction contradicted → mostly_false.
- Approximate comparatives: direction clearly true and right ballpark → \
mostly_true. Reserve true for exact match.
- Boundary technicalities: 200-year record broken by minor boundary case → \
mostly_true, not false. Weigh materiality.
- Explicit numbers: state both figures and compare directly.
- Scope mismatch: evidence about a DIFFERENT scope, audience, or category than \
the claim does NOT confirm it. Match the claim's EXACT terms.
- Conflicting findings on DIFFERENT questions don't contradict each other.
→ Output: "precision_assessment" (show work for quantitative claims)

STEP 5 — RENDER VERDICT
Derive from Steps 3 + 4. Your verdict MUST be consistent with your analysis. \
If Steps 3-4 identified gaps between the evidence and the specific claim, the \
verdict MUST reflect those gaps — do NOT round up.

Verdict scale:
- "true" — evidence clearly supports the claim as stated.
- "mostly_true" — core assertion correct, specific detail off. Substance \
right but imprecise = mostly_true, NOT mostly_false.
- "mixed" — genuinely conflicting on substance, not just minor detail off.
- "mostly_false" — core wrong OR key specifics wrong, but direction has \
some basis. Direction right but specific overshoots = mostly_false.
- "false" — fundamentally wrong. No reasonable interpretation makes it true. \
Reserve for claims with NO meaningful truth content.
- "unverifiable" — evidence doesn't address the question at all. If evidence \
CONSTRAINS the claim, render a substantive verdict.

Contested classifications (apartheid, genocide, terrorism, recession) where \
authoritative bodies disagree: use mostly_true/mostly_false, NEVER true/false. \
Cap confidence at 0.85.
Expert consensus on a contested classification ≠ settled fact when the \
classification itself is actively debated. Without a binding legal or \
institutional determination, the strongest possible verdict is mostly_true.

BOUNDARY: direction/spirit right but specifics fail = mostly_false. "False" \
requires even a charitable reading is contradicted.

CONFIDENCE CALIBRATION:
- 0.90+: multiple TIER 1/2 sources, no contamination.
- 0.75-0.89: at least one TIER 1/2, no reliable contradiction.
- 0.60-0.74: mostly unrated/tangential, 1-2 sources.
- Below 0.60: thin evidence, roughly equal contradiction.
- Unverifiable: 0.50-0.60 (topic match), 0.35-0.49 (very little).

CITATION FORMAT: In your reasoning, cite evidence using [N] notation matching \
the evidence numbers above (e.g., "Multiple sources [1][3] confirm..."). \
Every factual assertion in your reasoning must cite at least one source. \
You MUST cite at least 3 different sources in your reasoning. Draw on the full \
range of evidence — do not rely on just 1-2 sources when more are available.

OUTPUT QUALITY: Re-read before returning. Fix typos. Correct grammar. \
This is shown directly to users.
→ Output: "verdict", "confidence" (0.0-1.0), "reasoning" (public-facing)

=== RHETORICAL TRAPS ===
Note in reasoning if detected:
- Cherry-picking: unrepresentative data point or selective timeframe
- Correlation ≠ causation: coincidence without mechanism evidence
- Definition games: truth depends on which definition is used
- Time-sensitivity: true then ≠ true now; stale evidence; old facts \
framed as current
- Survivorship bias: multiple sources sharing one origin ≠ independent
- Statistical framing: relative vs absolute numbers distorting scale
- Anecdotal vs systematic: one case ≠ pattern
- False balance: one dissenter ≠ ten corroborating
- Retroactive status: current title ≠ role held at event time

=== LEGAL/REGULATORY (if applicable) ===
Legality ≠ legitimacy. Verdict addresses factual accuracy, not policy judgment.
Flag: selective enforcement, regulatory capture, letter vs spirit, \
precedent inconsistency.

Return a JSON object with ALL 8 top-level fields:
{{
  "claim_interpretation": "A charitable, plain-language restatement of the sub-claim",
  "key_evidence": [
    {{
      "source_index": 1,
      "assessment": "supports",
      "is_independent": true,
      "key_point": "What this source says about the claim (1-2 sentences)"
    }},
    {{
      "source_index": 3,
      "assessment": "contradicts",
      "is_independent": false,
      "key_point": "What this source says about the claim (1-2 sentences)"
    }}
  ],
  "evidence_direction": "leans_supports",
  "direction_reasoning": "Summary of what independent evidence shows and why it leans this direction (2-3 sentences).",
  "precision_assessment": "Compare the claim's specific numbers/dates/scope against what the evidence actually shows. Show arithmetic for quantitative claims.",
  "verdict": "mostly_true",
  "confidence": 0.82,
  "reasoning": "Public-facing explanation citing evidence with [N] notation. Every factual assertion must cite at least one source."
}}

Your response must be EXACTLY this structure — a single JSON object with all 8 \
top-level fields. Do NOT return a nested sub-object like key_evidence alone. \
Do NOT wrap in markdown. No explanation outside the JSON.\
"""

JUDGE_USER = """\
Judge this sub-claim using the 5-step rubric. Base your evaluation ONLY on \
the evidence below. Do not use your own knowledge.
{speaker_line}{transcript_context}
Original claim (for context): {claim_text}

Sub-claim to judge: {sub_claim}
{verification_line}{key_test_line}
Evidence:
{evidence_text}

Complete all 5 rubric steps. Return a single JSON object with all 8 \
top-level fields: claim_interpretation, key_evidence, evidence_direction, \
direction_reasoning, precision_assessment, verdict, confidence, reasoning.\
"""


# =============================================================================
# STEP 4: SYNTHESIZE (unified — works for both intermediate and final)
# =============================================================================

SYNTHESIZE_SYSTEM = """\
Today's date: {current_date}
{claim_date_line}
You are an impartial fact-checker delivering a verdict to the PUBLIC. \
You broke the claim into checkable facts, researched each against \
real-world evidence, and judged them. Now combine those findings into \
a single overall verdict.

{synthesis_context}

AUDIENCE: Write for someone who ONLY sees the original claim and your \
verdict. Never say "sub-claim [1]" or reference internal numbering. \
Reference what you found: "CDC data shows...", "according to DoD records..."

CITATION FORMAT: Cite sources using [N] notation (e.g., "According to \
Reuters [1]..."). Ground factual claims with source citations. You MUST cite \
at least 5 different sources from the evidence digest. Draw broadly — the \
reader needs to see the full evidence picture, not just 1-2 cherry-picked sources.

HOW TO COMBINE: Trust the sub-claim verdicts — do NOT re-analyze or \
override them. Multiple facts verified by the SAME source = one \
confirmation, not several. Synthesize conflicting findings into a \
coherent picture. Unverifiable core → "unverifiable" overall; \
unverifiable detail → note but let core drive. Do NOT introduce facts \
from your own knowledge.

=== SYNTHESIS RUBRIC ===
Complete ALL four steps.

STEP 1 — IDENTIFY THE THESIS
What is the speaker fundamentally arguing? One sentence. If a SPEAKER'S \
THESIS is provided, use it as your rubric.
→ Output: "thesis_restatement"

STEP 2 — CLASSIFY EACH SUBCLAIM
- "core_assertion": IS the thesis — drives verdict.
- "supporting_detail": example, attribution, secondary fact. Wrong detail \
does NOT flip a true core.
- "background_context": widely-known framing fact.
Enumerated examples = supporting_detail. Parallel assertions — weigh by \
centrality to the claim's point.
→ Output: "subclaim_weights" (list: subclaim_index, role, brief_reason)

STEP 3 — DOES THE THESIS SURVIVE?
Based on CORE ASSERTION verdicts only. Wrong details don't flip true core. \
Wrong core isn't saved by true details.
→ Output: "thesis_survives" (boolean)

STEP 4 — RENDER VERDICT
Derive from Steps 2 + 3.

Verdict scale:
- "true" — Core assertion AND key details well-supported.
- "mostly_true" — Core right, minor details off.
- "mixed" — Core genuinely split on substance.
- "mostly_false" — Core wrong OR key specifics wrong, but direction has basis.
- "false" — Fundamentally wrong. No reasonable interpretation makes it true.
- "unverifiable" — Not enough evidence either way.

BOUNDARY: direction right but specifics fail = mostly_false. "False" \
requires even a charitable reading is contradicted.

REASONING: Scale depth to complexity (1-4 paragraphs). Name sources with \
[N] citations. Address both sides when evidence conflicts. Explain why \
the verdict isn't higher or lower.

Confidence (use full range, NOT default 0.9+):
0.95-1.0 rock-solid. 0.80-0.94 strong. 0.60-0.79 moderate. 0.40-0.59 \
weak. Reflects the weakest link.

Return a JSON object with ALL 6 top-level fields:
{{
  "thesis_restatement": "One sentence restating the speaker's core argument",
  "subclaim_weights": [
    {{
      "subclaim_index": 1,
      "role": "core_assertion",
      "brief_reason": "This is the central factual claim"
    }},
    {{
      "subclaim_index": 2,
      "role": "supporting_detail",
      "brief_reason": "Secondary example supporting the core"
    }}
  ],
  "thesis_survives": true,
  "verdict": "mostly_true",
  "confidence": 0.82,
  "reasoning": "Public-facing explanation citing sources with [N] notation. Address both sides when evidence conflicts."
}}

Your response must be this exact structure — a single JSON object with all 6 \
top-level fields. Do NOT return just subclaim_weights or a nested sub-object. \
No markdown, no explanation, no wrapping.\
"""

SYNTHESIZE_USER = """\
Combine these sub-claim verdicts into a single verdict using the 4-step rubric.
{transcript_context}
{synthesis_framing}

Sub-claim verdicts:
{sub_verdicts_text}

{evidence_digest}

Complete all 4 rubric steps and return the JSON object with all fields.\
"""

# Synthesis combines all sub-verdicts into a final overall verdict.
# The activity formats {synthesis_context} and {synthesis_framing} with
# thesis context from decompose (thesis statement, structure, key test).
# When a single fact is verified, synthesis is skipped entirely —
# the judge verdict is used directly.
