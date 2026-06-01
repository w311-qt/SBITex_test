# LLM-as-Judge: Architecture, Limitations, Recommendation

## 1. Judge Architecture

```
question ──────────────────────────────────────────────┐
reference_answer ──────────────────────────────────────┤
                                                        ▼
                                              ┌─ correctness judge ─┐
                                              │  system: rules for  │
answer + context ─────────────────────────────┤  negative cases,    │──► {score, label, reason}
                                              │  invented symbols   │
                                              └─────────────────────┘
answer + context ────────────────────────────►┌─ faithfulness judge ─┐
  (no reference)                              │  context-only;       │──► {score, label, reason}
                                              │  invented symbols    │
                                              │  = hallucination     │
                                              └──────────────────────┘
question + answer ───────────────────────────►┌─ relevance judge ───┐
                                              │  on-topic check     │──► {score, label, reason}  (bonus)
                                              └─────────────────────┘
                                                        │
                                           dual-judge: same calls via
                                           secondary model (llama3.2:3b)
                                           → AgreementResult{score_delta, label_agree}
                                           → Cohen's κ over full dataset
```

**Primary judge** — same model and endpoint as the RAG generator (Qwen, OpenAI-compatible API).
Temperature = 0. Retry ≤ 2 on JSON parse failure. Falls back to `score=0.0, label="parse_error"`.

**Two prompt strategies:**

| Dimension | Input | What it measures |
|---|---|---|
| Correctness | question + reference + candidate | Semantic agreement with gold answer |
| Faithfulness | question + context + candidate | Grounding in retrieved chunks; no reference visible |

A third optional **Relevance** judge (question + candidate, on-topic check) runs per case and is aggregated as `mean_relevance` — a bonus metric reported alongside correctness/faithfulness (0.660 BM25-only → 0.819 hybrid).

**Negative-case handling in correctness:** The reference describes the *correct denial* ("WinMerge does not use CUDA…") rather than being left empty, giving the judge a concrete target. A candidate that hallucinates a positive contradicts that reference → correctness near 0.0. In practice the generator instead refuses ("not present in the retrieved context"); since that refusal does not reproduce the reference's specific denial, the judge scores it low as well (corr 0.0–0.25). That is acceptable: `false_answer_rate` (spec §6) flags a negative case only when correctness ≥ 0.5, so correct refusals — well below the threshold — are never miscounted as false answers.

**Dual-judge:** A second call goes to `llama3.2:3b` (a different model family, via the same Ollama endpoint). Labels are binarised at 0.5 and Cohen's κ is computed over the full dataset. κ > 0.6 indicates reliable agreement; lower values flag cases requiring human review.

---

## 2. LLM-as-Judge Limitations

**Self-evaluation bias.** When the generator and judge share the same model, the judge tends to approve its own style of reasoning. This inflates faithfulness and correctness scores for outputs the model "likes." Mitigated here by using a different family (`llama3.2:3b`) as the second judge and comparing κ.

**Verbosity bias.** Judges reliably score longer, more confident answers higher, regardless of factual accuracy. This causes false negatives for correct short answers and false positives for fluent hallucinations.

**Position / recency bias.** The judge gives higher scores to content appearing earlier in long prompts. Long context chunks can mask the actual candidate.

**Hallucination of the judge itself.** On rare inputs the judge invents reasons that contradict its own score. The `label` field (coarse-grained) and `reason` field (free text) often disagree in edge cases — worth logging separately.

**τ sensitivity.** `pass_rate` and derived boolean metrics depend heavily on the chosen threshold. A fixed τ = 0.7 may be miscalibrated for a new model or domain. The `calibrate` command addresses this with 5+ human labels.

**Refusal misclassification (small-model judge).** A 7B judge following a multi-rule faithfulness prompt inconsistently applies rules — notably, pure refusals ("not in context") can receive faithfulness=0.3 instead of the correct 1.0, which would inflate `hallucination_rate` with a false positive. Mitigation applied: (1) faithfulness is forced to 1.0 for structurally-detected refusals (< 40 words, no new technical symbols), removing the false positive; (2) `detected_refusal_rate` exposes the underlying retrieval-failure count independent of judge calibration. Correctness, by contrast, scores refusals low (corr ≈ 0.0–0.1), so `false_refusal_rate` (is_refusal AND corr < 0.3) catches them correctly — 0.233 on BM25-only → 0.033 on hybrid as retrieval improves.

**Language asymmetry.** Multilingual prompts produce lower agreement between models that differ in their multilingual training. Russian questions evaluated by an English-dominant model may get systematically lower faithfulness scores.

**BM25 definition vs. usage bias.** BM25 rewards files that *use* a class most frequently, not the file that *defines* it. `DiffWrapper.cpp` outranks `DiffList.h` for "DiffList class" queries because it mentions DiffList in dozens of call sites. Mitigation: per-file chunk deduplication + multiplicative stem-match bonus (file name tokens matching query tokens signal a definition file).

**Hybrid retrieval (BM25 + dense).** Pure BM25 cannot match Russian questions to English code, nor a symbol name (DIFFRANGE) to its declaring file (DiffList.h) — the tokens don't overlap. We add a dense retriever (`intfloat/multilingual-e5-base`, cosine over chunk embeddings) and fuse with Reciprocal Rank Fusion. Two findings drove the design:
1. *Project-name poisoning:* every question contains "WinMerge", which tokenizes to {win, merge}; "merge" spuriously matched the `Merge.*` file stems and granted them the definition boost on every query. Fixed with a stem stopword list.
2. *Noise dilution under RRF:* for Russian queries BM25 returns pure corpus-frequency noise (top-1 score ≈ 8.9, vs ≥16 for any real lexical match). Equal-weight RRF let that noise dilute the (weak but correct) dense signal, so hybrid scored no better than BM25 alone. Fix: **confidence-gated fusion** — BM25 joins the fusion only when its top-1 score clears a floor (15), otherwise the query is served dense-only. Stable across floor ∈ [12, 18].

Result (final dataset): evidence_recall 0.767 (BM25-only) → 0.900 (gated hybrid) — this is the BM25-only vs hybrid A/B shown in the README (composite 0.706 → 0.775). Remaining 3 Russian misses (`wm_ru_02/04/05`) need a stronger multilingual embedder (bge-m3) or query translation; e5-base does not rank them top-5.

---

## 3. What to Implement Instead of Keyword Eval

The current keyword-based eval (evidence files + forbidden terms + `should_have_answer` flag) is deterministic and cheap — keep it as a fast CI gate. The semantic layer should complement it, not replace it.

**Immediate upgrade: LLM-as-judge** (this project).
Captures whether the *meaning* is correct, not just whether a keyword appears. A model that answers "the diff engine is in `CDiffWrapper`" passes keyword checks but may still hallucinate the method name.

**Mid-term: claim-level verification (RAGAS-style).**
Decompose the reference into atomic claims ("CDiffWrapper uses WinMergeDiffItem", "RunFileDiff launches the diff thread", …). Check each claim independently in the candidate answer. This gives fine-grained recall/precision per fact, not a holistic score. Reduces verbosity bias because each claim is short.

**Mid-term: embedding similarity baseline.**
BERTScore or cosine similarity over sentence embeddings gives a cheap non-LLM signal. Use it as a sanity check: if LLM judge score is high but BERTScore is low, flag the case for human review.

**Long-term: swap-augmented judging.**
Run the judge twice — once with (reference, candidate) and once with (candidate, reference) swapped — and average scores. Eliminates order bias. Cost: 2× judge calls.

**Pipeline:** `keyword gate (fast, CI) → LLM judge (semantic, nightly) → human spot-check on low-agreement cases (weekly)`.
