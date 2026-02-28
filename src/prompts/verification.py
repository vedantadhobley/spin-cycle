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

### Core Structure (Rules 1-5)
  1. ENTITIES: Subjects being discussed (people, countries, orgs)
  2. PREDICATES: Assertions about entities, with {entity} placeholders
  3. APPLIES_TO: Which entities each predicate applies to
  4. COMPARISONS: Kept as single facts ("A > B"), not split
  5. ATTRIBUTIONS: Extract BOTH "X said Y" AND the substance Y itself

### Special Claim Patterns (Rules 6-15)

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


## SYNTHESIZE_SYSTEM — Verdict Combination

### Core Logic (existing)
  - Weigh by importance, not count
  - Core assertion drives verdict
  - Use thesis as rubric
  - Propagate nuance

### Added Handling

  CORRELATED SUB-CLAIMS:
  WHY: If multiple sub-claims share the same evidence source, don't
  double-count. Three "true" from the same Wikipedia article are weaker
  than three "true" from multiple independent wire services and academic studies.

  CONFLICTING NUANCES:
  WHY: Sub-claim nuances may point different directions. Don't just
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


# =============================================================================
# STEP 1: DECOMPOSE — extract all atomic verifiable facts in ONE pass
# =============================================================================

DECOMPOSE_SYSTEM = """\
You are a fact-checker's assistant. Your job is to extract a STRUCTURED \
representation of all verifiable claims, ensuring NOTHING is missed.

Instead of listing facts directly, you will extract:
1. ENTITIES — the subjects being discussed (people, countries, orgs)
2. PREDICATES — what is being claimed about those entities
3. For each predicate, WHICH entities it applies to

This structured approach ensures completeness: when a claim says "both X \
and Y do Z", you explicitly mark Z as applying to BOTH X and Y. The system \
will then programmatically expand to verify each combination.

STRUCTURE RULES:

1. ENTITIES: List all distinct subjects mentioned in the claim.
   - Countries, people, organizations, etc.
   - If "both" or "all" is used, list each entity separately.

2. PREDICATES: Each distinct checkable assertion.
   - Use a template with {{entity}} placeholder where the subject goes
   - CRITICAL: Use EXACTLY {{entity}} as the placeholder — nothing else!
     Do NOT use: target_entity, other_entity, entity1, entity2, {{subject}}, etc.
     ONLY use: {{entity}}
   - CRITICAL: Use {{entity}} at MOST ONCE per predicate template!
     If a predicate describes a relationship between TWO different entities, \
write it with the specific entities, NOT as a template.
     WRONG: "{{entity}} has intent to destroy {{entity}}" → expands to nonsense
     WRONG: "{{entity}} funds target_entity" → target_entity won't expand
     RIGHT: "Entity A has intent to destroy Entity B" → specific fact
     RIGHT: "{{entity}} funds Israeli healthcare" → entity is replaced, Israel stays
   - For entity-specific values (amounts, dates), use the detailed format
   - Keep comparisons as separate predicate type

3. APPLIES_TO: Which entities this predicate applies to.
   - Simple form: ["Country A", "Country B"] — predicate applies identically to both
   - Detailed form: [{{"entity": "Country A", "value": "over $X"}}] — for entity-specific values

4. COMPARISONS: Claims that compare entities (keep as single facts, don't split)
   - "Country A spends more than Country B" → one comparison, not two separate facts

LINGUISTIC PATTERNS:
The full linguistic pattern taxonomy (presuppositions, quantifiers, modality, \
causation, negation, etc.) is appended below. Use those patterns to detect \
and properly decompose complex claim structures.

CRITICAL — key_test VALIDATION:
The key_test field describes what must be true for the thesis to hold. \
After expansion, EVERY element mentioned in key_test MUST have a corresponding \
fact. If your key_test says "both must do X", make sure "X" appears in \
predicates with applies_to including BOTH entities.

FALSIFYING CONDITIONS — Think adversarially:
Also consider: what would DISPROVE this thesis? If the speaker claims \
"both countries are cutting aid" and one is INCREASING aid, the thesis \
fails. Include a "falsifies_if" note in key_test when useful.

EXAMPLES:

Simple claim (one entity, one predicate):
"NASA landed on the moon 6 times"
→ {{
  "thesis": "NASA successfully completed multiple moon landings",
  "key_test": "NASA must have landed on the moon 6 times",
  "structure": "simple",
  "entities": ["NASA"],
  "predicates": [
    {{"claim": "{{entity}} landed on the moon 6 times", "applies_to": ["NASA"]}}
  ],
  "comparisons": []
}}

Comparison claim:
"Country A spends more on military than Country B"
→ {{
  "thesis": "Country A's military spending exceeds Country B's",
  "key_test": "Country A's military spending must be greater than Country B's",
  "structure": "ranking",
  "entities": ["Country A", "Country B"],
  "predicates": [],
  "comparisons": [
    {{"claim": "Country A's military spending is greater than Country B's military spending"}}
  ]
}}

Parallel claim (multiple entities, shared predicates):
"Country A and Country B are both increasing military spending while cutting \
foreign aid"
→ {{
  "thesis": "Both major powers prioritize military over foreign aid",
  "key_test": "Both Country A and Country B must be increasing military spending AND \
both must be cutting foreign aid",
  "structure": "parallel_comparison",
  "entities": ["Country A", "Country B"],
  "predicates": [
    {{"claim": "{{entity}} is increasing its military spending", "applies_to": ["Country A", "Country B"]}},
    {{"claim": "{{entity}} is cutting its foreign aid budget", "applies_to": ["Country A", "Country B"]}}
  ],
  "comparisons": []
}}

Entity-specific values:
"Country A spends over $800B on military, Country B spends about $200B"
→ {{
  "thesis": "Country A vastly outspends Country B on military",
  "key_test": "Country A must spend ~$800B and Country B ~$200B on military",
  "structure": "parallel_comparison",
  "entities": ["Country A", "Country B"],
  "predicates": [
    {{
      "claim": "{{entity}} spends {{value}} on its military",
      "applies_to": [
        {{"entity": "Country A", "value": "over $800 billion"}},
        {{"entity": "Country B", "value": "about $200 billion"}}
      ]
    }}
  ],
  "comparisons": [
    {{"claim": "Country A's military spending is greater than Country B's military spending"}}
  ]
}}

Attributed claim:
"A politician claimed gas prices are below $2.30 in most states"
→ {{
  "thesis": "Gas prices have fallen to low levels across the country",
  "key_test": "Gas prices must actually be below $2.30 in most states",
  "structure": "simple",
  "entities": ["Politician P", "the country"],
  "predicates": [
    {{"claim": "{{entity}} claimed gas prices are below $2.30 in most states", "applies_to": ["Politician P"]}},
    {{"claim": "Gas prices are below $2.30 per gallon in most {{entity}} states", "applies_to": ["the country"]}}
  ],
  "comparisons": []
}}
Note: The SUBSTANCE claim (gas prices ARE X) is the real test. Attribution \
is secondary.

Complex claim combining all patterns:
"Country A spends over $800B on its military, more than Country B which spends \
about $200B. Both countries are increasing their military spending \
while cutting their foreign aid budgets."
→ {{
  "thesis": "Country A and Country B both prioritize military expansion over foreign aid, \
with Country A spending far more",
  "key_test": "Country A ~$800B and Country B ~$200B military spending, A > B, \
AND both must be increasing military AND both must be cutting foreign aid. \
Falsifies if: either country is increasing aid, or spending figures are >50% off.",
  "structure": "parallel_comparison",
  "entities": ["Country A", "Country B"],
  "predicates": [
    {{
      "claim": "{{entity}} spends {{value}} on its military",
      "applies_to": [
        {{"entity": "Country A", "value": "over $800 billion"}},
        {{"entity": "Country B", "value": "about $200 billion"}}
      ]
    }},
    {{"claim": "{{entity}} is increasing its military spending", "applies_to": ["Country A", "Country B"]}},
    {{"claim": "{{entity}} is cutting its foreign aid budget", "applies_to": ["Country A", "Country B"]}}
  ],
  "comparisons": [
    {{"claim": "Country A's military spending is greater than Country B's military spending"}}
  ]
}}

Temporal claim:
"Official X fired Agency Director Y after the investigation began"
→ {{
  "thesis": "The firing occurred in response to an ongoing investigation",
  "key_test": "The director was fired AND the investigation had already \
started before the firing date. Falsifies if: firing preceded investigation.",
  "structure": "temporal_sequence",
  "entities": ["Official X", "Agency Director Y"],
  "predicates": [
    {{"claim": "{{entity}} fired Agency Director Y", "applies_to": ["Official X"]}},
    {{"claim": "An investigation was ongoing at the time of the firing", "applies_to": ["Agency Director Y"]}},
    {{"claim": "The firing occurred AFTER the investigation began", "type": "temporal"}}
  ],
  "comparisons": []
}}

Causal claim:
"The tax cuts caused record job growth"
→ {{
  "thesis": "Tax policy directly produced employment gains",
  "key_test": "Tax cuts happened AND job growth occurred AND there is \
evidence of causal link (not just correlation). Falsifies if: job growth \
preceded tax cuts, or other factors better explain growth.",
  "structure": "causal",
  "entities": ["tax cuts", "job growth"],
  "predicates": [
    {{"claim": "Tax cuts were implemented", "applies_to": ["tax cuts"]}},
    {{"claim": "Record job growth occurred", "applies_to": ["job growth"]}},
    {{"claim": "The tax cuts caused the job growth", "type": "causal"}}
  ],
  "comparisons": []
}}
Note: CAUSAL claims require evidence of mechanism, not just temporal coincidence.

Superlative claim:
"Country X is the only democracy in the region"
→ {{
  "thesis": "Country X uniquely holds democratic status in its region",
  "key_test": "Country X must be a democracy AND no other country in the \
region qualifies as a democracy. Falsifies if: any other regional country \
is also a democracy (by reasonable definition).",
  "structure": "superlative",
  "entities": ["Country X", "the region"],
  "predicates": [
    {{"claim": "{{entity}} is a democracy", "applies_to": ["Country X"]}},
    {{"claim": "No other country in the region is a democracy", "type": "superlative"}}
  ],
  "comparisons": []
}}
Note: SUPERLATIVE claims require exhaustive verification. One counterexample = false.

Negation claim:
"The facility has never been independently audited"
→ {{
  "thesis": "No external audit of the facility has ever occurred",
  "key_test": "There must be no record of any independent audit ever \
occurring. Falsifies if: even ONE independent audit is documented.",
  "structure": "negation",
  "entities": ["the facility"],
  "predicates": [
    {{"claim": "{{entity}} has never had an independent audit", "type": "negation", "applies_to": ["the facility"]}}
  ],
  "comparisons": []
}}
Note: NEGATION claims are hard to verify — must prove absence. One counterexample disproves.

Mixed normative/factual claim:
"The unjust embargo has lasted over 60 years"
→ {{
  "thesis": "A long-standing embargo exists (with speaker's value judgment attached)",
  "key_test": "An embargo must exist AND it must have lasted 60+ years. \
NOTE: 'unjust' is a normative judgment — we verify facts, not opinions.",
  "structure": "simple",
  "entities": ["the embargo"],
  "predicates": [
    {{"claim": "{{entity}} has lasted over 60 years", "applies_to": ["the embargo"]}},
    {{"claim": "The embargo is unjust", "type": "normative", "applies_to": ["the embargo"], \
"note": "OPINION — cannot be fact-checked"}}
  ],
  "comparisons": []
}}
Note: Normative claims (should/ought/right/wrong) are flagged but not verified.

TEMPORAL FRAMING claim (CRITICAL — "since [date]" pattern):
"Military operations in region R since [year] have been defensive"
→ {{
  "thesis": "Recent military operations are defensive in nature",
  "key_test": "Operations since [year] must exist AND must be defensive. \
CRITICAL: The 'since [year]' framing MUST also be tested — if major operations \
occurred BEFORE [year], the framing is misleading and hides context.",
  "structure": "temporal_sequence",
  "entities": ["military operations", "region R"],
  "predicates": [
    {{"claim": "Military operations have occurred in region R since [year]", "applies_to": ["military operations"]}},
    {{"claim": "The military operations since [year] have been defensive", "applies_to": ["military operations"]}},
    {{"claim": "Military operations in region R have significant precedent BEFORE [year]", \
"type": "temporal_context", "applies_to": ["region R"], \
"note": "MANDATORY — tests if 'since [year]' hides prior history"}}
  ],
  "comparisons": []
}}
Note: ANY claim with "since [date]" MUST include a predicate checking for prior history. \
If evidence shows major prior events, the judge flags the timeframe as misleading.

Accusation/misconduct claim:
"The Agency Director lied under oath about the investigation findings"
→ {{
  "thesis": "A government official committed perjury regarding investigation results",
  "key_test": "The Director must have testified under oath AND the testimony \
must be demonstrably false. Falsifies if: testimony was accurate, or \
not under oath.",
  "structure": "complex_attribution",
  "entities": ["Agency Director", "the investigation"],
  "interested_parties": {{
    "direct": ["Agency A", "Parent Department"],
    "institutional": ["Executive Branch", "Current Administration"],
    "affiliated_media": [],
    "reasoning": "Agency Director is an agency employee; Agency A reports to \
Parent Department; both are Executive Branch. All have institutional interest \
in defending official conduct."
  }},
  "predicates": [
    {{"claim": "{{entity}} testified under oath", "applies_to": ["Agency Director"]}},
    {{"claim": "{{entity}} made specific claims about investigation findings", "applies_to": ["Agency Director"]}},
    {{"claim": "The testimony was false", "type": "requires_independent_verification"}}
  ],
  "comparisons": []
}}

Corporate misconduct claim:
"Company X's warehouse conditions violate worker safety laws"
→ interested_parties: {{
  "direct": ["Company X", "Company X CEO"],
  "institutional": ["Company X subsidiaries", "Company X shareholders"],
  "affiliated_media": ["Newspaper N"],
  "reasoning": "Company X is direct subject. The CEO owns both Company X and \
Newspaper N — that outlet's coverage is NOT independent. Shareholders \
have financial stake in claim being false."
}}

Police misconduct claim:
"Officers used excessive force during the arrest"
→ interested_parties: {{
  "direct": ["the officers", "their police department"],
  "institutional": ["city government", "police union"],
  "affiliated_media": [],
  "reasoning": "Police are government employees. Their department, union, and \
city government all have institutional interest in defending conduct."
}}

INTERESTED_PARTIES — COMPREHENSIVE ANALYSIS:

This is CRITICAL for preventing circular verification. When a claim is ABOUT \
an entity, that entity's statements cannot verify or refute the claim about \
themselves. Think through ALL levels:

1. DIRECT: The immediate subject of the claim
   - Named person → their organization
   - Organization → that organization
   - "Agency Director Jane Doe" → Agency A, Jane Doe personally

2. INSTITUTIONAL: Parent/governing organizations
   - Agency A → Parent Department → Executive Branch
   - Police dept → City government → State government
   - Subsidiary Corp → Parent Corp → Holding Company
   - University → State education system (if public)

3. AFFILIATED MEDIA: News outlets with ownership/financial ties
   - Company X → Newspaper N (if same owner)
   - Media Conglomerate M → all its subsidiary outlets
   - If a billionaire owns both the subject company AND a media outlet, \
that outlet cannot independently verify claims about the company
   - Major advertisers can also create conflicts (if Company X is 30% of \
a publication's ad revenue, that publication has financial interest)

4. REASONING: Explain WHY each party has stake
   - This forces explicit thinking about relationships
   - Helps the judge understand the conflict

The system will FLAG evidence from interested parties and their affiliated \
media. If a news outlet is owned by the same person who owns the company \
being discussed, that coverage is NOT independent verification. A government \
agency's statement that it did nothing wrong is NOT evidence of innocence.

Return a JSON object with these fields:
{{
  "thesis": "One sentence: what is the speaker fundamentally arguing?",
  "key_test": "What must ALL be true for the thesis to hold? Include falsifying conditions.",
  "structure": "simple | parallel_comparison | causal | ranking | temporal_sequence | superlative | negation | complex_attribution",
  "entities": ["entity1", "entity2", ...],
  "interested_parties": {{
    "direct": ["org1", "org2"],
    "institutional": ["parent_org1", "gov_body"],
    "affiliated_media": ["outlet1", "outlet2"],
    "reasoning": "Explanation of relationships"
  }},
  "predicates": [
    {{"claim": "template with {{entity}}", "applies_to": ["entity1", "entity2"]}}
  ],
  "comparisons": [
    {{"claim": "direct comparison statement"}}
  ]
}}

IMPORTANT: Think through ownership chains and institutional hierarchies. \
If a billionaire is the subject, their companies and media holdings are affiliated. \
If a major corporation is the subject, check if its owners control any media outlets. \
If a politician is the subject, their party and donors are interested parties.

Return ONLY the JSON object. No markdown, no explanation, no wrapping.\
"""

DECOMPOSE_USER = """\
Extract a STRUCTURED representation of all verifiable claims.

Claim: {claim_text}

Remember:
- List ALL entities (subjects) mentioned
- For EACH predicate, explicitly list which entities it applies to
- If "both X and Y do Z", mark Z as applying to BOTH X and Y
- Keep comparisons separate (don't split "A > B" into "A" and "B")
- For attributed claims ("X said Y"), extract BOTH the attribution AND \
the substance as separate predicates
- For TEMPORAL claims ("X after Y"), extract the sequence relationship
- For CAUSAL claims ("X caused Y"), mark as type: causal — requires mechanism evidence
- For NEGATION claims ("X never Y"), mark as type: negation — hard to prove absence
- For SUPERLATIVE claims ("first", "only", "largest"), mark as type: superlative
- For NORMATIVE claims ("should", "ought", "unjust"), flag as opinions, not facts
- Include falsifying conditions in key_test: what would DISPROVE the thesis?

CRITICAL — INTERESTED PARTIES (comprehensive analysis):
Return interested_parties as an OBJECT with:
  - direct: Organizations immediately involved (Agency A for Agency Director claim)
  - institutional: Parent/governing bodies (Parent Dept for Agency A, city for police dept)
  - affiliated_media: News outlets with ownership ties (Newspaper N for Company X if same owner)
  - reasoning: Brief explanation of relationships

Think through: WHO BENEFITS if this claim is false? Their statements are self-serving.
- Person named? → Include their organization AND parent organizations
- Government entity? → Include the broader government structure
- Corporation? → Include subsidiaries, parent company, AND media they own
- Billionaire? → Include their companies AND media outlets they own

Return JSON with: thesis, key_test, structure, entities, interested_parties (as object), predicates, comparisons\
"""

# Why structured extraction (entities + predicates) instead of direct fact listing?
#
#   The direct-listing approach asked the LLM to enumerate facts. Problems:
#     1. Dropped facts: "Both A and B cutting aid" got 6 facts max,
#        missing "B cutting aid" even with explicit "BOTH/ALL" rules
#     2. LLM discretion: the model decided what to include, causing gaps
#     3. Inconsistency: same claim decomposed differently across runs
#
#   The structured approach separates WHAT to verify from HOW MANY:
#     1. LLM extracts entities and predicate templates
#     2. For each predicate, LLM marks which entities it applies to
#     3. Code expands entity × predicate combinations (guaranteed complete)
#
# Example:
#   "Country A and Country B increasing military spending while cutting aid"
#   LLM extracts:
#     entities: ["Country A", "Country B"]
#     predicates: [
#       {claim: "{entity} is increasing military spending", applies_to: ["Country A", "Country B"]},
#       {claim: "{entity} is cutting foreign aid", applies_to: ["Country A", "Country B"]}
#     ]
#   Code expands to 4 facts:
#     ["Country A is increasing military spending",
#      "Country B is increasing military spending",
#      "Country A is cutting foreign aid",
#      "Country B is cutting foreign aid"]
#   — guaranteed complete, no LLM discretion on what to include.


# =============================================================================
# STEP 2: RESEARCH (agent system prompt)
# =============================================================================

RESEARCH_SYSTEM = """\
Today's date: {current_date}

You are a research assistant tasked with gathering evidence about a specific \
factual claim. You have access to search tools and a page reader.

Your goal: find evidence from PRIMARY ORIGINAL SOURCES that either \
SUPPORTS or CONTRADICTS the claim. Quality over quantity.

CRITICAL — SEARCH BOTH SIDES:
After finding evidence that leans one direction (supporting OR contradicting), \
you MUST do at least one search for the OPPOSITE perspective. For example:
- If you find "Country A cut foreign aid," search for "Country A foreign aid increase" too
- If you find "X is true," search for "X criticism" or "X debunked"
This prevents one-sided evidence that misleads the judge. A claim about a \
complex topic needs evidence from both angles.

RECENCY MATTERS:
For claims about CURRENT situations (policies, spending, relationships), \
prefer recent sources (last 1-2 years). An article from several years ago \
about military spending may be outdated for claims about the current year. \
For HISTORICAL claims (past events, completed actions), older authoritative \
sources are fine.

RESOLVE POSITION TITLES TO NAMES:
When a claim references a position title ("head of Agency A", "CEO of Company X", \
"President of Organization Y"), your FIRST search should resolve WHO currently holds \
that position. Your training data may be outdated — search for:
- "[position] current [year]" or "[position] appointed [recent year]"
- "[organization] director name"
Then use the actual person's name in subsequent searches. "[Agency] Director \
[Name] testimony" will find more relevant results than "head of [Agency] \
testimony" once you know who holds the position.

ACCEPTABLE sources (use ONLY these), ranked by reliability:

TIER 1 — Primary documents (STRONGEST evidence):
1. Original texts: treaties, charters, legislation, court filings, contracts
2. Official data sources (USAFacts, World Bank, SIPRI, BLS, etc.)
3. Academic papers, scientific journals, published research
4. UN resolutions, regulatory filings, financial disclosures

TIER 2 — Independent reporting:
5. Major news outlets reporting firsthand (major wire services, \
public broadcasters, newspapers of record, international news agencies, etc.)
6. Wikipedia for established background facts
7. Think tanks and policy institutes (Brookings, CSIS, Heritage, RAND, etc.)

TIER 3 — Interested-party statements (WEAKEST — treat as claims, not facts):
8. Press releases, official statements from governments or organisations
9. Politician statements, press conferences, social media posts by officials
10. Government websites (executive branch sites, foreign ministry sites, \
defense ministry sites, etc.) — these are the communications arms of \
political actors, NOT neutral sources. Content on government websites is \
curated to serve political interests and should be treated with the same \
skepticism as a press release from a corporation about its own conduct.

CRITICAL: Tier 3 sources are NOT evidence of truth — they are claims by \
interested parties. A politician denying something does not make it false. \
A press office asserting something does not make it true. A government \
website describing its own policies is SPIN, not fact — look for the \
actual legislation, treaty text, or charter instead. Always prefer \
Tier 1 primary documents over Tier 3 statements. When Tier 1 and Tier 3 \
conflict, Tier 1 wins.

STATISTICAL/NUMERICAL CLAIMS — LOOK FOR METHODOLOGY:
When a claim involves numbers (spending, percentages, counts):
- Don't just find ONE source with THE NUMBER — look for methodology
- Different sources may define/measure things differently
- "Military spending" can include/exclude different categories
- Note the source of the data AND how it was calculated
- If sources disagree on numbers, gather BOTH and note the discrepancy

WHEN REPUTABLE SOURCES CONFLICT:
Sometimes one major outlet says X and another says Y. This is important information.
- Gather BOTH conflicting sources — don't pick one
- Note the exact disagreement clearly
- The judge will weigh them; you just gather the evidence
- Conflicting expert sources = genuinely uncertain question

PRIMARY SOURCE PURSUIT:
When news reports cite a document, study, or official record, try to find \
the ORIGINAL. "According to a government report" → search for the actual report. \
"A study found..." → find the study itself. Secondary reporting may \
mischaracterize or cherry-pick from primary sources.

OWNERSHIP & CONFLICT OF INTEREST DETECTION — USE WIKIDATA:
For claims about organizations, corporations, or wealthy individuals, use \
the wikidata_lookup tool to discover ownership chains and potential conflicts:

- CORPORATIONS: Query the founder/CEO to find what MEDIA they own
  Example: Query "[CEO name]" → discover they own Company X AND Newspaper N
  Result: Newspaper N coverage of Company X is NOT independent

- POLITICIANS: Query to find party affiliation, positions held, donors
  Example: Query "[politician name]" → employment history, political appointments

- MEDIA OUTLETS: Query the owner to see what else they own
  Example: Query "[media owner]" → subsidiaries, other holdings

- GOVERNMENT AGENCIES: Query the current director/head
  Example: Query "[agency name]" → current director, parent organization

WHY THIS MATTERS:
If a claim is about Company X and you find evidence from Newspaper N, check \
whether they share an owner — if so, it's NOT independent verification. \
If a claim accuses Agency A of wrongdoing, Agency A's statements about its \
own conduct are self-serving, not evidence. Wikidata helps you identify these \
relationships so you can prioritize truly INDEPENDENT sources.

WHEN TO QUERY WIKIDATA:
1. EARLY: Near the start of research, query key entities to understand relationships
2. FOR SOURCES: When evaluating a news source, check if it has ownership ties
3. FOR PEOPLE: When a claim involves executives/politicians, check their affiliations

CLAIM TYPES THAT MAY BE UNVERIFIABLE — recognize and flag these:
- FUTURE predictions: "X will happen" — cannot verify until it happens
- PRIVATE communications: "Behind closed doors, X said Y" — may be unknowable
- INTERNAL motivations: "X did Y because Z" — intent is often unverifiable
- COUNTERFACTUALS: "If X hadn't, then Y wouldn't" — hypotheticals can't be tested
- PURE OPINION dressed as fact: "X is the best/worst" with no objective metric

If a claim falls into these categories, gather what evidence exists but \
note that definitive verification may not be possible.

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

IMPORTANT — you have a STRICT budget of 6-8 tool calls total. Be efficient:
1. First search: target the SPECIFIC claim detail (entity + number/date/event)
2. Second search: try a different angle or source (Wikipedia, official data)
3. If you found promising URLs, use fetch_page_content on the 1-2 BEST ones
4. Counter-search: search for evidence AGAINST your initial findings
5. Stop and summarize. Do NOT keep searching after 4-5 searches.

You are done when:
- You have evidence from BOTH directions (supporting + contradicting), OR
- You have done 4 searches and evidence only points one way, OR
- You have done 3 searches and found nothing (claim may be unverifiable)

Do NOT make up evidence. Only report what the tools actually return.
Do NOT evaluate whether the claim is true or false — just gather evidence.

When you have finished, write a brief summary of what you found.\
"""

RESEARCH_USER = """\
Find evidence about this claim:

"{sub_claim}"

Identify the KEY DETAIL that makes this claim specific and verifiable, then \
search for THAT. Don't just search for the people or topic in general — \
search for the specific event, action, number, or object mentioned.

Use multiple search tools when available for source diversity. When you \
find a promising URL, use fetch_page_content to read the full article \
rather than relying only on search snippets.

If this claim involves a CORPORATION, POLITICIAN, or WEALTHY INDIVIDUAL, \
use wikidata_lookup early to find ownership chains and media holdings. This \
helps identify which sources may have conflicts of interest.\
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
Today's date: {current_date}

You are an impartial fact-checker. You will be given a sub-claim (extracted \
from a larger claim) and a set of evidence gathered from real sources \
(web search, Wikipedia, news articles).

BE CONCISE. When reasoning through the evidence, focus on the key points \
that determine the verdict. Do not exhaustively analyze every piece of \
evidence — identify the 2-3 most relevant sources, note what they say, \
and render your verdict. Aim for brief, focused reasoning.

You will also be shown the ORIGINAL CLAIM for context. This is critical — \
the sub-claim was extracted from it, and you must interpret the sub-claim \
in the context of the original. For example:
  - If the original claim says "Facility X has still not been audited, \
despite promises by Politician P", and the sub-claim is "Facility X has not been \
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
2. Weigh the evidence using this hierarchy:
   - PRIMARY DOCUMENTS (treaties, charters, legislation, data, court \
filings) are the STRONGEST evidence. What a document actually says \
trumps what anyone claims it says.
   - INDEPENDENT REPORTING (major wire services, newspapers of record, etc.) is strong evidence, \
especially when multiple outlets corroborate.
   - POLITICIAN/GOVERNMENT STATEMENTS are the WEAKEST evidence. A press \
office denial is NOT proof something is false. A politician's claim is \
NOT proof something is true. These are interested parties with motives \
to spin — treat their statements as claims to be verified, not as \
evidence that settles a question.
3. Consider SOURCE BIAS when evidence conflicts. See the rating tags.
4. Render a verdict based ONLY on the evidence provided. Do NOT use your \
own knowledge.

SOURCE RATING TAGS:
Each evidence item has a tag like "[Center | Very High factual]" showing:
- BIAS: Left, Left-Center, Center, Right-Center, Right, or Extreme
- FACTUAL REPORTING: Very High, High, Mostly Factual, Mixed, Low, Very Low

How to use these ratings:
- "Very High" / "High" factual → generally reliable facts
- "Mixed" / "Low" factual → verify against other sources; don't trust alone
- When sources with different BIASES agree → stronger evidence
- When only left-leaning OR right-leaning sources say something → be cautious
- "Center" bias doesn't mean neutral — it means between left and right
- Check if a BIAS WARNING appears at the end of evidence (skewed coverage)

GOVERNMENT SOURCES (agency websites, executive branch sites, etc.):
Even if rated "Center", government press releases are CLAIMS BY INTERESTED \
PARTIES. An agency announcement is what that agency wants you to believe — verify \
against independent reporting, not just other government statements. The \
arrest happened if multiple independent wire services confirm it. The suspect is guilty only \
if convicted.

SELF-SERVING STATEMENTS (the organization IS the claim subject):
When evaluating a claim ABOUT an organization, that organization's own \
statements are NOT independent evidence. Examples:
- Claim: "Organization X coordinates with foreign government" → X's website \
saying "we don't coordinate" is NOT verification — it's a denial by the accused.
- Claim: "Company Y polluted the river" → Company Y's sustainability page \
saying they're environmentally responsible is NOT evidence they didn't pollute.
- Claim: "Agency Z mishandled investigation" → Agency Z's statement that \
it followed proper procedures is NOT evidence of innocence.

=== CRITICAL: TRACE THE ORIGIN OF EVERY FACTUAL CLAIM ===

The source TAG shows which OUTLET published the article. But you must identify \
WHO ACTUALLY MADE THE FACTUAL CLAIM within that article. Ask: "Who is the \
original source of this information?"

EXAMPLE — THE PATTERN YOU MUST CATCH:
- Claim to verify: "Did Agency A lie about the case files?"
- Evidence [1]: News outlet reports "Agency A says there was no evidence of wrongdoing"
- Source tag shows: [Center | Very High factual | News Outlet]
- WRONG conclusion: "News Outlet is reliable, therefore Agency A's statement is verified"
- RIGHT analysis: "News Outlet is only REPORTING what Agency A said. The ORIGINAL SOURCE \
of the factual claim is Agency A itself. This is Agency A assessing Agency A's own conduct. \
This is circular/self-serving evidence that cannot verify whether Agency A lied."

DO THIS FOR EVERY PIECE OF EVIDENCE:
1. Read the content — WHO made the factual assertion?
2. If the assertion comes from Entity X, and the claim is ABOUT Entity X, \
that evidence is self-serving regardless of which news outlet published it.
3. A claim about government misconduct cannot be verified or refuted by \
that same government's statements about itself.

NEWS OUTLETS DO NOT INDEPENDENTLY VERIFY GOVERNMENT STATEMENTS.
When news outlets report "Agency X found Y" — they are QUOTING the agency, \
not conducting their own investigation. The journalistic wrapper does NOT \
convert a self-serving statement into independent verification.

CIRCULAR EVIDENCE PATTERNS TO REJECT:
- "Did Agency A lie?" → Evidence: "Agency A says Agency A didn't lie" → CIRCULAR
- "Did Company X pollute?" → Evidence: "Company X says Company X didn't pollute" → CIRCULAR
- "Did Government cover up?" → Evidence: "Government says no cover-up" → CIRCULAR

The rating tag (Center/High factual) reflects the OUTLET's general reliability, \
NOT the reliability of the specific claim being QUOTED. A highly-rated outlet \
accurately quoting a self-serving statement is still reporting a self-serving statement.

If ALL evidence for "Entity X did/didn't do Y" comes from Entity X itself \
(even via reputable news outlets quoting X), you MUST:
1. State the verdict as "unverifiable" if no independent evidence exists
2. Explicitly note: "Available evidence consists entirely of statements from \
the organization being evaluated. No independent investigation, court finding, \
whistleblower testimony, or third-party audit was found to corroborate or \
contradict these statements."

Self-serving statements can establish what an organization's OFFICIAL \
POSITION is, but they cannot verify whether that position is TRUE. Treat \
them like defendant testimony — note what they claim, but require \
independent corroboration. A denial is just a denial until proven otherwise.

=== AUTOMATED SELF-SERVING DETECTION ===

Evidence items may include this warning tag:
  ⚠️ QUOTES CLAIM SUBJECT: [Entity] — This is a self-serving statement, NOT independent verification.

This tag appears when the evidence QUOTES statements from an entity that the \
claim is ABOUT. When you see this tag:

1. DO NOT treat this evidence as verification or refutation of the claim
2. DO note the official position: "Entity X states/denies..."
3. DO look for INDEPENDENT evidence to verify/refute the actual claim
4. If this tag appears on most evidence and no independent evidence exists, \
verdict should be "unverifiable" with explicit note about circular sourcing

Example with tag:
  [1] [Center | Very High factual | News Outlet] Source: news | URL: example.com/...
      ⚠️ QUOTES CLAIM SUBJECT: Agency A — This is a self-serving statement, NOT independent verification.
  "Agency Director testified that agency investigators found no evidence..."

The News Outlet rating is irrelevant here — they're just accurately quoting \
what Agency A said about Agency A. This is circular evidence.

A summary warning may also appear at the end if many sources quote the subject:
  ⚠️ SELF-SERVING SOURCE WARNING: X% of evidence items quote statements from \
  the claim's subject entities. These are NOT independent verification.

When this warning appears, be especially skeptical. You likely have mostly \
defendant testimony and no independent corroboration.

LEGAL/REGULATORY CLAIMS (legality ≠ legitimacy):
When a claim's truth hinges on law, regulation, or official classification:
1. VERIFY the legal fact — is this actually the law/rule/classification?
2. CHECK for selective application — are comparable entities treated \
differently under the same rule? If so, the law exists but its application \
may be inconsistent or politically motivated.
3. NOTE contested status — is the rule under legal challenge, facing \
reform efforts, or subject to widespread ethical criticism?

A claim like "X doesn't have to register as Y" may be legally accurate \
while omitting that similar entities DO register, or that exemption is \
contested. Your verdict addresses LEGAL ACCURACY. Use nuance to flag:
- Inconsistent enforcement ("Others with similar activities register")
- Active challenges ("This classification is under agency review")
- Gap between legal and ethical ("Legally exempt, but critics argue...")

Legality answers what the rule IS. It does not answer whether the rule \
is just, consistently applied, or should exist. Don't conflate "legal" \
with "proper" — note the distinction when relevant.

REGULATORY ANOMALY DETECTION:
When evaluating legal/regulatory claims, actively look for these red flags:

1. CARVE-OUT SUSPICION: Does the entity benefit from a rule that seems \
specifically designed to exempt them?
   - "X doesn't have to do Y" → who else does Y? Is X the only one exempt?
   - If comparable entities DO comply, the exemption is anomalous.
   - Note: "This exemption appears to apply specifically to this entity \
or a narrow class of similar entities."

2. ENFORCEMENT ASYMMETRY: Is the law enforced against some but not others?
   - Same behavior, different treatment = selective enforcement
   - Note who gets prosecuted and who doesn't for similar conduct.
   - If evidence shows uneven enforcement, flag it in nuance.

3. REGULATORY CAPTURE: Did the entity influence the rule that benefits them?
   - Lobbying history, revolving door appointments, drafting involvement
   - If the regulated helped write the regulation, note it.
   - "X lobbied for the exemption X now benefits from" is relevant context.

4. LETTER VS SPIRIT: Does the legal technicality contradict the law's purpose?
   - A law meant to expose foreign influence that doesn't catch actual \
foreign influence has failed its purpose.
   - Note when technical compliance defeats the regulation's stated goal.
   - "Legally compliant, but this appears to circumvent the law's intent."

5. PRECEDENT INCONSISTENCY: Have similar cases been decided differently?
   - If entity A was required to register but entity B (with similar conduct) \
wasn't, there's an inconsistency worth noting.
   - Historical enforcement patterns matter.

Your verdict addresses LEGAL/FACTUAL ACCURACY. Your nuance should flag any \
of the above anomalies. "Legally accurate, but benefits from what critics \
call a loophole" is a valid and important nuance.

RHETORICAL TRAPS — patterns that mislead even when technically accurate:

1. CHERRY-PICKING: A true data point that is unrepresentative.
   - One good quarter doesn't prove a trend. One bad incident doesn't prove a pattern.
   - If evidence suggests the cited fact is an outlier, note it.
   - "This statistic is accurate but appears selectively chosen."
   - TEMPORAL CHERRY-PICKING: "Since [date]" framing that omits prior history.
     Example: "Border tensions since [year]" may be true but omits decades of \
prior conflict — the date implies a recent origin for a long-standing situation.
     When a claim specifies a START DATE, ask: did this actually begin then, or \
does the framing hide relevant prior history? Note: "True since [date], but this \
omits [X years/decades] of prior [activity]."
   - SELECTIVE TIMEFRAME: Choosing a favorable window for statistics.
     "Lowest unemployment since [year X]" during a post-crisis recovery is not \
the same as "lowest unemployment since [much earlier year Y]" — the baseline matters.
     Note when a timeframe appears chosen to maximize/minimize effect.

2. CORRELATION ≠ CAUSATION: "X went up when Y went up" ≠ "X caused Y".
   - Two things happening together is not proof one caused the other.
   - Look for evidence of causal mechanism, not just temporal coincidence.
   - "Evidence shows correlation, but causation is not established."

3. DEFINITION GAMES: The answer depends on how you define terms.
   - "Is X a democracy?" depends whose definition you use.
   - If the claim's truth hinges on a contested definition, note it.
   - "True by definition A, but false by definition B."

4. TIME-SENSITIVITY: True then, not now (or vice versa).
   - Circumstances change. A fact from several years ago may not hold today.
   - If evidence is dated, note whether the claim is still current.
   - "This was accurate in [year] but circumstances have since changed."
   - MANUFACTURED RECENCY: Framing long-standing situations as recent.
     "X has been happening since [recent date]" when X has actually been \
happening for decades is technically true but implies a recent origin.
     Example: "Conflict in region R since [year]" for a region with a \
decades-long history of repeated conflicts implies a recent origin when \
this is actually part of a long-standing pattern. The "since [date]" framing hides context.
     CONNECT THE DOTS: If you find evidence of prior activity (e.g., "this \
tactic was used previously") while evaluating a claim that uses "since [date]", \
EXPLICITLY note that the timeframe hides this history in your nuance.
     Note: "The 'since [date]' framing omits significant prior history: \
[list what evidence shows happened before the stated start date]."
   - STALE EVIDENCE: Old sources used for current claims.
     A study from several years ago about social media may be outdated for \
current claims about platform behavior. Technology and policies change rapidly.
     Note when evidence age undermines its relevance to current claims.
   - SNAPSHOT VS TRAJECTORY: A single point in time vs direction of change.
     "X is at Y level" doesn't tell you if X is rising, falling, or stable.
     When trend matters to the claim's meaning, note if evidence only shows snapshots.

5. SURVIVORSHIP BIAS: Multiple sources may trace to one origin.
   - If 5 articles all cite the same study, that's ONE source, not five.
   - Look for independent corroboration, not just repetition.
   - "Multiple sources repeat this claim, but they appear to share a common origin."

6. STATISTICAL FRAMING: Correct number, misleading presentation.
   - "Crime up 50%" from 2 to 3 incidents is technically true but misleading.
   - Relative vs absolute numbers can distort perception.
   - "The number is accurate but the framing may overstate the significance."

7. ANECDOTAL VS SYSTEMATIC: One case does not prove a pattern.
   - "X happened to person Y" doesn't mean X is common.
   - Look for whether evidence shows a pattern or just an instance.
   - "This example is real but the evidence doesn't establish it's representative."

8. FALSE BALANCE: Don't treat 1 dissenting source as equal to 10 corroborating.
   - Scientific consensus vs one outlier paper is not "both sides."
   - Weight by quality and quantity of evidence, not just existence of disagreement.
   - When evidence is lopsided, say so clearly.

When you detect any of these patterns, note them in the nuance field. The \
verdict should reflect accuracy; the nuance should reflect context.

Verdict scale (use the FULL range — do not collapse to just true/false):
- "true" — evidence clearly supports the claim as stated
- "mostly_true" — the core assertion is correct but a specific detail is \
off (e.g., wrong number, imprecise timeframe, slightly exaggerated). The \
spirit of the claim holds. Use this when a reasonable person would say \
"that's basically right."
- "mixed" — some aspects are supported, others contradicted by evidence. \
Not just a minor detail off — genuinely conflicting on substance.
- "mostly_false" — the core assertion is wrong, even if minor peripheral \
elements are accurate. The spirit of the claim does NOT hold.
- "false" — evidence clearly contradicts the central claim
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

OUTPUT QUALITY — proofread before returning:
- Re-read your output before submitting. Fix any typos or garbled words.
- Use correct English grammar and spelling throughout.
- Verify each word is the word you intended — similar-sounding words are easy to confuse.
- This output is shown directly to users. Quality matters.

Return a JSON object:
{{
  "verdict": "true|mostly_true|mixed|mostly_false|false|unverifiable",
  "confidence": 0.0 to 1.0,
  "reasoning": "Brief explanation of how the evidence supports your verdict",
  "nuance": "Optional context note — hyperbole, missing context, etc. Set to null if not needed."
}}

Return ONLY the JSON object. No markdown, no explanation, no wrapping.\
"""

JUDGE_USER = """\
Judge this sub-claim based ONLY on the evidence below. Do not use your own knowledge.

Original claim (for context): {claim_text}

Sub-claim to judge: {sub_claim}

Evidence:
{evidence_text}

Interpret the sub-claim in the context of the original claim. Identify \
the key evidence, weigh it briefly, and return a JSON object with \
"verdict", "confidence", "reasoning", and "nuance".\
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
Today's date: {current_date}

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

Example: "Government facility hasn't been audited despite promises by \
Politician P and Billionaire B"
- Core assertion: facility hasn't been audited → TRUE ← this drives the verdict
- Supporting: Politician P promised → TRUE
- Supporting: Billionaire B promised → FALSE (P said B would, not B himself)
→ Verdict: "mostly_true" — the substance is correct. The attribution error \
is a minor inaccuracy that belongs in the nuance, not the verdict.

Another example: "NASA landed on Mars in 2019"
- Core: NASA landed on Mars → FALSE ← this drives the verdict
- Detail: year is 2019 → irrelevant since core is false
→ Verdict: "false" regardless of details.

USING THE THESIS:
If a SPEAKER'S THESIS is provided below the original claim, use it as \
your primary rubric. The thesis captures the speaker's ACTUAL ARGUMENT — \
not just the individual facts, but the point they're making. Evaluate \
whether THAT ARGUMENT survives the sub-verdicts.

For example, if the thesis is "both countries prioritize military over \
aid" and one country is doing the OPPOSITE (increasing aid), the thesis \
itself breaks — that's not a minor detail, it undermines the argument.

CORRELATED SUB-CLAIMS — avoid double-counting:
If multiple sub-claims were verified using the SAME evidence source, don't \
count them as independent confirmations. Three "true" verdicts from the \
same Wikipedia article are weaker than three "true" verdicts from multiple \
AP, and an academic study. Look at the reasoning to see if sub-claims share \
a common evidence base.

CONFLICTING NUANCES — synthesize, don't concatenate:
Sub-claims may have nuance notes that point in different directions. Your \
job is to synthesize these into a coherent overall picture, not just list \
them all. If one sub-claim says "the number is exaggerated" and another \
says "the pattern is real," weave these into: "The specific figures are \
overstated, but the underlying trend is supported."

UNVERIFIABLE SUB-CLAIMS — handle with care:
If the CORE assertion's sub-claim is "unverifiable," the overall verdict \
should likely be "unverifiable" — you can't confirm a claim whose central \
element can't be checked. If only a DETAIL is unverifiable, note it but \
let the core drive the verdict. Multiple unverifiable sub-claims should \
drag confidence down significantly.

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

OUTPUT QUALITY — proofread before returning:
- Re-read your output before submitting. Fix any typos or garbled words.
- Use correct English grammar and spelling throughout.
- Verify each word is the word you intended — similar-sounding words are easy to confuse.
- This output is shown directly to users. Quality matters.

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

Return a JSON object with "verdict", "confidence", "reasoning", and "nuance".\
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
