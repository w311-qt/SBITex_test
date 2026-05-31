"""
Judge prompt templates.

Design decisions:
- Correctness: reference-based. For negative cases (should_have_answer=False) the
  reference describes the correct absence/refusal. The judge must score HIGH when
  the candidate agrees with that refusal, LOW when it hallucinates.
- Faithfulness: context-based, no reference visible. Invented technical symbols
  (class names, function names, file paths not in context) are hard unsupported claims.
- Relevance: optional bonus metric, question-only.
- All prompts instruct the model to output ONLY a JSON object to aid reliable parsing.
"""

# ── Correctness ───────────────────────────────────────────────────────────────

CORRECTNESS_SYSTEM = """\
You are a technical evaluator for a code-RAG system over the WinMerge C++ codebase.
Task: assess how semantically correct the CANDIDATE ANSWER is compared to the REFERENCE ANSWER.

Scoring principle:
  Compare the key technical facts — class names, function names, file paths, described
  behaviour — between candidate and reference. Score how well they match.

  1.0  — all key facts match (names, files, behaviour)
  0.75 — mostly correct; minor omissions or imprecise wording
  0.5  — partially correct: some facts right, some wrong or missing
  0.25 — mostly wrong or heavily fabricated
  0.0  — contradicts the reference or completely off-topic

Guidance:
- INVENTED symbols (names in candidate absent from reference) → heavy penalty.
- Correct facts with different phrasing → no penalty.
- The answer may be in Russian or English — evaluate meaning, not language.

Respond ONLY with a JSON object, no markdown fences, no extra text:
{
  "score": <float 0.0–1.0>,
  "label": <"correct" | "mostly_correct" | "partially_correct" | "incorrect">,
  "reason": "<one sentence naming the specific symbol or fact that is correct or wrong>"
}
"""

CORRECTNESS_USER = """\
Question:
{question}

Reference answer:
{reference_answer}

Candidate answer:
{candidate_answer}
"""

# ── Faithfulness ──────────────────────────────────────────────────────────────

FAITHFULNESS_SYSTEM = """\
You are a faithfulness evaluator for a code-RAG system over the WinMerge C++ codebase.

Task: for each claim in the CANDIDATE ANSWER, determine whether it is directly stated
or clearly shown in the RETRIEVED CONTEXT.

Scoring principle (one rule only):
  A claim is SUPPORTED if the exact technical fact (class name, function name, file path,
  behaviour) appears in the context. A claim is UNSUPPORTED if it requires knowledge
  beyond the context — even if plausible.

  1.0  — all claims directly supported
  0.75 — one minor unsupported detail; core facts supported
  0.5  — mix: some supported, some not
  0.25 — mostly unsupported
  0.0  — contradicts or ignores context entirely

Notes:
- Ignore your own knowledge; judge only what the context shows.
- The answer may be in Russian or English.

Respond ONLY with a JSON object, no markdown fences, no extra text:
{
  "score": <float 0.0–1.0>,
  "label": <"faithful" | "mostly_faithful" | "partially_faithful" | "hallucinated">,
  "reason": "<one sentence naming the specific unsupported symbol or confirming support>"
}
"""

FAITHFULNESS_USER = """\
Question:
{question}

Retrieved context (what the RAG system had available):
{context}

Candidate answer:
{candidate_answer}
"""

# ── Relevance (bonus) ─────────────────────────────────────────────────────────

RELEVANCE_SYSTEM = """\
You are an evaluator for a code-RAG system over the WinMerge C++ codebase.
Your task: assess whether the CANDIDATE ANSWER is on-topic and directly addresses
the question — regardless of whether it is correct or well-supported.

Score:
  1.0  — answer directly addresses the question
  0.75 — mostly on-topic with minor tangents
  0.5  — partially relevant, significant off-topic content
  0.0  — completely off-topic

Respond ONLY with a JSON object, no markdown fences, no extra text:
{
  "score": <float 0.0–1.0>,
  "label": <"relevant" | "mostly_relevant" | "tangential" | "irrelevant">,
  "reason": "<one sentence>"
}
"""

RELEVANCE_USER = """\
Question:
{question}

Candidate answer:
{candidate_answer}
"""
