"""Comprehensive linguistic pattern taxonomy for claim decomposition.

This module contains the canonical patterns from formal semantics, pragmatics,
and rhetoric that affect how claims should be decomposed and verified.

These patterns are based on established linguistic literature, not ad-hoc
rules discovered through trial and error. The LLM was trained on this
literature, so using proper terminology triggers better pattern recognition.

References:
- Levinson (1983) "Pragmatics" — presupposition, implicature
- Portner (2009) "Modality" — modals, evidentials
- Horn (1989) "A Natural History of Negation" — negation scope
- Kratzer (2012) "Modals and Conditionals" — modal semantics
- Grice (1975) "Logic and Conversation" — conversational implicature
- Google SAFE (NeurIPS 2024) — claim decomposition for fact-checking
- FActScore (Min et al. 2023) — atomic fact extraction
"""

# =============================================================================
# LINGUISTIC PATTERN TAXONOMY
# =============================================================================

LINGUISTIC_PATTERNS = """

## LINGUISTIC PATTERNS FOR CLAIM VERIFICATION

When decomposing claims, detect and properly handle these linguistic patterns.
Each pattern affects what sub-claims must be extracted and how they should be verified.


### 1. PRESUPPOSITION TRIGGERS

Presuppositions are hidden assumptions embedded in word choice. The claim takes
them for granted without explicitly stating them.

**Trigger words and their presuppositions:**
| Trigger | Example | Presupposition |
|---------|---------|----------------|
| "stopped", "quit", "ceased" | "He stopped lying" | He was lying before |
| "again", "another", "return" | "War broke out again" | There was war before |
| "still", "continue", "remain" | "Still denies it" | Was denying it before |
| "started", "began", "initiated" | "Started operations" | No operations before |
| "before", "after", "since" | "Since the attack" | The attack happened |
| "regret", "realize", "know" | "Regrets the decision" | Made the decision |
| "only", "even", "just" | "Only 3 countries" | Expectation of more |

**How to handle:**
1. IDENTIFY the trigger word
2. EXTRACT the presupposition as an explicit sub-claim
3. EXTRACT the asserted content as a separate sub-claim
4. Both must be verified — a false presupposition makes the whole claim misleading

**Example:**
"Israel started operations in Gaza due to October 7th"
→ Presupposition: "Israel had no significant operations in Gaza before October 7th"
→ Assertion: "Israel launched new operations in Gaza after October 7th"
→ Causal claim: "October 7th caused the new operations"

If the presupposition is false (Israel DID have operations before), the claim
is misleading even if the assertion is technically true.


### 2. QUANTIFIER SCOPE

Quantifiers dramatically change truth conditions. The difference between "all",
"most", and "some" is not stylistic — it's semantic.

**Quantifier strength (strongest to weakest):**
| Quantifier | Truth requires | One counterexample |
|------------|----------------|-------------------|
| all, every, each, always | 100% of cases | → claim is FALSE |
| most, majority, usually | >50% of cases | → still TRUE |
| many, often, frequently | substantial portion | → ambiguous |
| some, several, sometimes | at least one case | → still TRUE |
| few, rarely, seldom | small portion | → ambiguous |
| none, never, no one | 0% of cases | One example → FALSE |

**How to handle:**
1. PRESERVE the exact quantifier — don't paraphrase "most" as "many"
2. Tag the claim with quantifier type so judge knows the threshold
3. For strong quantifiers (all/none), one counterexample is decisive
4. For vague quantifiers (many/few), note the ambiguity


### 3. MODALITY

Modal verbs change whether a claim is an assertion of fact or something weaker.

**Epistemic modals (about possibility/likelihood):**
| Modal | Strength | Verifiability |
|-------|----------|---------------|
| must, certainly | Strong assertion | Verifiable |
| will, going to | Prediction | NOT verifiable (future) |
| should, ought to | Expectation | Partially verifiable |
| may, might, could | Possibility | Often not verifiable |
| possibly, perhaps | Weak possibility | Often not verifiable |

**Deontic modals (about obligation/permission):**
| Modal | Type | Example |
|-------|------|---------|
| must, have to | Obligation | "Countries must comply" |
| should, ought to | Recommendation | "Should reduce emissions" |
| may, can | Permission | "May request an extension" |

**How to handle:**
1. Distinguish ASSERTIONS from POSSIBILITIES from OBLIGATIONS
2. "X may have done Y" is NOT "X did Y" — different claim type
3. Predictions (will/going to) are generally unverifiable


### 4. EVIDENTIALITY MARKERS

Words that indicate the source of information. Often used to hedge or
distance the speaker from the claim.

**Common evidentiality markers:**
| Marker | What it signals |
|--------|----------------|
| "reportedly", "allegedly" | Unverified report |
| "according to X" | Attributed to specific source |
| "sources say", "insiders claim" | Anonymous/vague attribution |
| "experts believe", "scientists say" | Appeal to authority (often vague) |
| "critics argue", "some say" | Dissenting view (which critics?) |
| "studies show", "research indicates" | Appeal to evidence (which studies?) |

**How to handle:**
1. IDENTIFY the evidentiality marker
2. EXTRACT: Who specifically is the source?
3. CHECK: Does the source actually say this?
4. CHECK: Is the substance of the claim actually true?
5. Vague attributions ("experts say") are often weasel words — flag them


### 5. TEMPORAL/ASPECTUAL

Time references affect what evidence is relevant and can hide context.

**Temporal framing issues:**
| Pattern | Problem | What to extract |
|---------|---------|-----------------|
| "since [date]" | Cherry-picked baseline | Test: was pattern different before date? |
| "in the last N years" | Arbitrary window | Test: does different window show different result? |
| "recently", "nowadays" | Vague timeframe | Clarify: what specific period? |
| "historically", "traditionally" | Vague past | Clarify: what specific period? |
| "unprecedented" | Requires exhaustive comparison | Test: any prior precedent? |

**Aspectual distinctions:**
| Aspect | Example | What it implies |
|--------|---------|-----------------|
| Perfective | "Built the wall" | Completed action |
| Progressive | "Building the wall" | Ongoing action |
| Habitual | "Builds walls" | Repeated pattern |
| Perfect | "Has built walls" | Past action, present relevance |

**How to handle:**
1. IDENTIFY temporal boundaries in the claim
2. EXTRACT explicit sub-claim testing the boundary: "Pattern X existed before [date]"
3. For ongoing situations: verify current status, not just past
4. For "unprecedented": must prove NO prior occurrence


### 6. CAUSATION TYPES

Causal claims vary in strength. Correlation is not causation.

**Causal hierarchy (strongest to weakest):**
| Type | Statement | Required evidence |
|------|-----------|-------------------|
| Sole cause | "X caused Y" | X is necessary AND sufficient for Y |
| Primary cause | "X mainly caused Y" | X is the largest factor |
| Contributing factor | "X contributed to Y" | X is one of several factors |
| Correlation | "X occurred with Y" | Pattern, not mechanism |
| Temporal sequence | "Y followed X" | Order, not causation |

**How to handle:**
1. PRESERVE exact causal language — "caused" vs "contributed to"
2. Sub-claim 1: Did X occur? (factual check)
3. Sub-claim 2: Did Y occur? (factual check)
4. Sub-claim 3: Is there a causal mechanism linking X to Y? (harder)
5. Correlation is NOT evidence of causation — flag this


### 7. COMPARISON AND DEGREE

Comparative and superlative claims have specific verification requirements.

**Comparison types:**
| Type | Example | What to verify |
|------|---------|----------------|
| Comparative | "A > B" | Both values, relationship |
| Superlative | "A is the most/least" | A's value + ALL alternatives |
| Equative | "A = B" | Both values, equality |
| Approximate | "roughly equal", "similar" | Both values, tolerance |

**Superlative triggers requiring exhaustive verification:**
- "first", "last", "only"
- "most", "least", "best", "worst"
- "largest", "smallest", "highest", "lowest"
- "unprecedented", "unique", "sole"

**How to handle:**
1. For superlatives: one counterexample makes the claim FALSE
2. Extract: "No X before [claimed first]" or "No X greater than [claimed most]"
3. Comparatives require BOTH values to be verified


### 8. NEGATION SCOPE

Negation claims (proving absence) are harder than positive claims.

**Negation types:**
| Type | Example | Verification difficulty |
|------|---------|------------------------|
| Existential negation | "Never happened" | Very hard — exhaustive search |
| Universal negation | "Nobody supports" | Very hard — check everyone |
| Specific negation | "X didn't do Y (on date Z)" | Easier — specific check |
| Constituent negation | "Not X but Y" | Medium — verify Y instead |

**How to handle:**
1. Flag negation claims explicitly — they need different research
2. "Never" claims: one counterexample disproves
3. Consider: is this claim practically verifiable? Some negations are unfalsifiable


### 9. SPEECH ACTS

Not all sentences are assertions. Only assertions are verifiable.

**Speech act types:**
| Type | Example | Verifiable? |
|------|---------|-------------|
| Assertion | "X is Y" | ✓ Yes |
| Question | "Is X Y?" | ✗ No (not a claim) |
| Command | "Stop X" | ✗ No (not a claim) |
| Promise | "I will do X" | ✗ Future, unverifiable |
| Opinion | "X is bad/good" | ✗ Subjective |
| Prediction | "X will happen" | ✗ Future, unverifiable |

**Normative vs Descriptive:**
| Type | Example | Verifiable? |
|------|---------|-------------|
| Descriptive | "X is illegal" | ✓ Yes (fact about law) |
| Normative | "X should be illegal" | ✗ Opinion |
| Mixed | "X is wrong" | Partially — factual aspect of "wrong" |

**How to handle:**
1. FILTER: Is this an assertion? If not, flag as uncheckable
2. SEPARATE: Mixed claims → extract factual parts, flag normative parts
3. Opinions are NOT verifiable — don't try


### 10. VAGUENESS AND HEDGING

Vague terms resist precise verification.

**Common vague terms:**
| Category | Examples |
|----------|----------|
| Quantity | "many", "few", "some", "significant", "substantial" |
| Frequency | "often", "sometimes", "rarely", "frequently" |
| Degree | "very", "extremely", "somewhat", "relatively" |
| Evaluation | "important", "major", "serious", "notable" |

**How to handle:**
1. IDENTIFY vague terms
2. If possible, CLARIFY with context: "many" = more than X?
3. Note vagueness in sub-claim: "Claim uses 'significant' without defining threshold"
4. Vague claims may be unverifiable due to ambiguity, not lack of evidence


### 11. REPORTED SPEECH / ATTRIBUTION

Claims about what someone said require double verification.

**Attribution types:**
| Type | Example | What to verify |
|------|---------|----------------|
| Direct quote | 'X said "Y"' | Did X say exactly Y? |
| Indirect report | "X said that Y" | Did X express Y? (paraphrase ok) |
| Implied attribution | "According to X, Y" | Does X's work support Y? |

**How to handle:**
1. Sub-claim 1: Did the attributed source actually say/write this?
2. Sub-claim 2: Is the substance (Y) actually true?
3. Substance is usually MORE important than attribution
4. Check for misquoting, out-of-context quotes


### 12. CONDITIONALS AND HYPOTHETICALS

Conditional claims have special verification needs.

**Conditional types:**
| Type | Example | Verifiable? |
|------|---------|-------------|
| Indicative | "If X happens, Y happens" | ✓ If X has happened |
| Counterfactual | "If X had happened, Y would have" | ✗ Usually not |
| Hypothetical | "If X were to happen, Y would happen" | ✗ Usually not |
| Predictive | "If X happens, Y will happen" | ✗ Future |

**How to handle:**
1. For indicative conditionals where X HAS occurred: verify Y
2. Counterfactuals are generally unverifiable — flag them
3. Extract both the condition (X) and consequence (Y) for separate verification


### 13. DEFINITION AND CATEGORY CLAIMS

"X is a Y" claims depend on how Y is defined.

**Common contested categories:**
- Genocide, war crime, terrorism, apartheid
- Democracy, dictatorship, authoritarianism
- Recession, depression, economic crisis
- Racism, discrimination, hate speech

**How to handle:**
1. IDENTIFY the category claim
2. EXTRACT: What definition is being used?
3. If contested: Note whose definition, and whether claim is true under alternatives
4. Legal definitions may differ from colloquial use


### 14. GENERICS

Generic statements ("Dogs bark") hide quantification.

**Generic types:**
| Type | Example | What it means |
|------|---------|---------------|
| Characterizing | "Dogs bark" | Typical/normal behavior |
| Kind reference | "The dinosaur is extinct" | About the kind, not individuals |
| Habitual | "John smokes" | Regular behavior |

**How to handle:**
1. Generics are technically false if ANY counterexample exists
2. But pragmatically, they're about typical/characteristic cases
3. Flag generics — they're often rhetorical rather than precise claims
4. "Politicians lie" isn't falsified by one honest politician


### 15. IMPLICATURE

What's implied but not stated. Pragmatic inference beyond literal meaning.

**Common implicatures:**
| Statement | Implicature |
|-----------|-------------|
| "Some students passed" | Not all passed |
| "The car is blue and old" | Blue and old are both relevant |
| "He's a good philosopher" | (Maybe not good at other things) |
| "She tried to solve it" | She failed |

**Scalar implicatures:**
- "Some" implies "not all"
- "Warm" implies "not hot"
- "Good" implies "not excellent"

**How to handle:**
1. Be aware that speakers may claim "I didn't say X" when X was implied
2. Extract BOTH literal meaning AND likely implicature
3. Note when speaker may be technically truthful but pragmatically misleading

"""


# =============================================================================
# DECOMPOSITION CHECKLIST
# =============================================================================

DECOMPOSITION_CHECKLIST = """

## DECOMPOSITION CHECKLIST

Before finalizing sub-claims, verify you have addressed ALL applicable patterns:

☐ **PRESUPPOSITIONS**: Did you extract hidden assumptions as explicit sub-claims?
   - Check for: started/stopped, again/still, before/after, only/even

☐ **QUANTIFIERS**: Did you preserve the exact quantifier?
   - "All" vs "most" vs "some" are DIFFERENT claims

☐ **MODALITY**: Is this an assertion or a possibility/prediction?
   - "May have" ≠ "did"; "will" = prediction (unverifiable)

☐ **EVIDENTIALITY**: Who is the source? Did you extract attribution?
   - "Experts say" → which experts?

☐ **TEMPORAL**: Did you test the timeframe boundaries?
   - "Since X" → what about before X?

☐ **CAUSATION**: Did you separate correlation from causation?
   - "X caused Y" needs causal mechanism, not just sequence

☐ **COMPARISONS**: For superlatives, did you extract exhaustive tests?
   - "First" → "No earlier instance exists"
   - "Only" → "No other instance exists"

☐ **NEGATION**: For "never"/"nobody" claims, did you note verification difficulty?

☐ **SPEECH ACTS**: Is this actually an assertion?
   - Opinions, predictions, and questions are not verifiable

☐ **VAGUENESS**: Did you flag undefined terms?
   - "Significant", "many", "often" without thresholds

☐ **ATTRIBUTION**: Did you extract BOTH who-said and what-they-said?

☐ **CONDITIONALS**: Is the condition fulfilled? Counterfactuals are unverifiable.

☐ **DEFINITIONS**: Is the category contested? Note whose definition.

☐ **GENERICS**: Is this a generic statement hiding quantification?

☐ **IMPLICATURE**: What does the claim IMPLY beyond what it states?

"""


# =============================================================================
# HELPER FUNCTION
# =============================================================================

def get_linguistic_patterns() -> str:
    """Return the complete linguistic pattern reference for prompts."""
    return LINGUISTIC_PATTERNS + DECOMPOSITION_CHECKLIST
