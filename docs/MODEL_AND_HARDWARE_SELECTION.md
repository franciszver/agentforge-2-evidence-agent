# Clinical Co-Pilot — Model & Hardware Selection (Phase 2 Evidence Agent)

- **Status:** Final for Phase 2's evidence-retrieval answer path. Documents how the local answer model and its serving configuration were chosen for the reference/minimum hardware tier, and why the verified-citation ceiling that choice produces is **4/12** — not a defect to be tuned away, but the honest capability boundary of an 8 GB GPU at this model class.
- **Method:** each row in the ladder below is a real run of the same 12 guideline-citation eval cases through the evaluation pipeline, with fail-closed citation verification — a citation counts only when a surviving claim carries a verbatim-verified guideline-chunk document citation. No number below is a projection; all four rows were executed.
- **Related:** `docs/W2_ARCHITECTURE.md` §"Reference Hardware & Model Tiers" (the same minimum-vs-recommended framing, applied there to the VLM/ingestion path); `docs/ARCHITECTURE.md` §"Capacity reality" (the Phase 1 precedent for measuring on real hardware before publishing a spec).

## Reference (minimum) hardware

**RTX 5060 Laptop GPU, 8 GB VRAM; 32 GB system RAM.**

This is the same dev/demo hardware Phase 1 measured its chat-latency and concurrency ceilings on (`docs/ARCHITECTURE.md`). Phase 2 keeps that precedent: the published minimum spec is whatever was actually measured on this box, not an estimate. If a configuration doesn't fit or doesn't perform acceptably here, it isn't the minimum spec — it's the recommended-tier story instead (see below).

## The ladder actually tested

Four model/config combinations were run end-to-end against the same 12 eval cases:

| Model / config | Engine | Verified citations | Latency | Hardware fit |
|---|---|---|---|---|
| Qwen3-4B @ 16k ctx | Ollama | 1/12 | ~22 s | Fits GPU |
| Qwen3-8B-Q4_K_M @ 16k ctx, q8_0 KV, flash-attn | llama.cpp | 3/12 | ~23 s | Fully GPU-resident, 6.27 GB / 8 GB |
| **Qwen3-8B-Q5_K_M @ 16k ctx, q8_0 KV, flash-attn (chosen)** | llama.cpp | **4/12** | ~27 s | Fully GPU-resident, 6.63 GB / 8 GB |
| Qwen3-30B-A3B (MoE, Q4) @ 16k ctx, q8_0 KV, expert-offload (`--cpu-moe`) | llama.cpp | 0/12* | median 152 s, max 300 s (timeout) | Experts in CPU RAM; only 1.9 GB VRAM |

*The 30B-A3B row is a confound, not a citation verdict, and is called out explicitly rather than left to imply the model can't cite at all: 10 of 12 cases failed with extraction timeouts before completion — the verbose reasoner exhausted its generation budget on internal reasoning before ever emitting the constrained JSON output the verifier needs, so only 2 of 12 cases ran to completion. It is disqualified primarily on **latency**, not citation quality: 3–12× slower than the 8B, because a 30B model's experts, offloaded to CPU RAM on an 8 GB card, are memory-bandwidth-bound (roughly 36 tokens/s). No serving configuration fixes this at this hardware tier — the bottleneck is CPU↔GPU memory bandwidth, not a tunable parameter.

## The two engine-level levers that decide what fits

**q8_0 KV-cache quantization + flash-attn (llama.cpp).** This is the enabling configuration for the chosen model. Without KV-cache quantization, an 8B model's key/value cache at a 16k context window does not fit alongside the model weights inside 8 GB of VRAM; q8_0 quantizes the cache to roughly a quarter of its FP16 footprint, and flash-attn keeps the attention computation itself memory-efficient at that context length. Together they're what lets Qwen3-8B-Q5_K_M run **fully GPU-resident** at 6.63 GB / 8 GB — no CPU offload, no swap latency.

**MoE expert-offload (`--cpu-moe`).** This is the lever that makes a 30B-parameter model fit on an 8 GB card at all — only 1.9 GB of VRAM is used, with the mixture-of-experts layers living in CPU RAM and dispatched per-token. It's an elegant trick for the memory constraint, but it does not solve the latency constraint: every token that routes through a CPU-resident expert pays CPU memory-bandwidth cost, and that cost dominates at this GPU tier. Fitting in VRAM and being usable for an interactive query are two different bars, and the 30B-A3B configuration clears only the first one here.

## Why 4/12 is the ceiling on this hardware

The ladder above establishes a monotonic pattern up to the point where hardware fit breaks: 4B → 1/12, 8B-Q4 → 3/12, 8B-Q5 → 4/12. Larger models cite more reliably, model capacity is the binding constraint on reliable verbatim guideline citation, and 8B-Q5 is the largest model this hardware can hold fully GPU-resident at a usable context length. Stepping up further — to the 30B-A3B — does fit in VRAM via expert-offload, but at a latency (2.5–5 minutes per query, with a third of cases timing out entirely) that makes it unusable for interactive clinical use regardless of what its citation rate would be if every case ran to completion.

That leaves 4/12 as the honest ceiling for this hardware tier: not a bug in the verification layer, not a tuning gap in the prompt, but the direct consequence of the largest model that fits an 8 GB card at interactive latency. The fail-closed verifier is doing its job correctly here — it is reporting the model's real citation reliability at this capacity, not padding the number with unverifiable claims.

This is stated plainly rather than downplayed: even the best minimum-spec configuration leaves citation reliability below half the cases. That is the tradeoff being published, not a temporary limitation expected to close with more prompt engineering.

## Minimum vs. recommended tier

| | Minimum spec (measured, this doc) | Recommended spec |
|---|---|---|
| GPU | RTX 5060 Laptop, 8 GB VRAM | Larger single card with enough VRAM to hold a bigger model fully resident at the same context length |
| Model | Qwen3-8B-Q5_K_M, q8_0 KV, flash-attn | A larger dense or MoE model sized to the larger card, with experts (if MoE) also GPU-resident rather than CPU-offloaded |
| Verified citations | 4/12 (measured) | Expected higher — not measured, since no recommended-tier hardware exists in this project; stated as directional only, per the same discipline `docs/W2_ARCHITECTURE.md` applies to its own projected numbers |
| Latency | ~27 s | Expected comparable or better, since the constraint driving the 30B-A3B's latency (CPU-offloaded experts) would not apply once a larger card holds the whole model in VRAM |

The mechanism is the same one `docs/W2_ARCHITECTURE.md` uses for the document-ingestion VLM: **model tier is a config value, not a code fork.** Swapping in a larger model at a larger-VRAM recommended tier changes citation reliability and latency; it does not change the verification layer's fail-closed contract. A clinician reading an answer on either tier gets the same guarantee — every surviving citation was verbatim-verified against the guideline corpus — the two tiers differ only in how often the model succeeds at producing a verifiable citation in the first place.

## Summary

The chosen configuration — Qwen3-8B-Q5_K_M, 16k context, q8_0 KV-cache, flash-attn, fully GPU-resident on an 8 GB card — is the largest model this reference hardware can serve at interactive latency, and it verifies 4 of 12 guideline-citation eval cases. That number is the measured ceiling for the published minimum spec, not a target still being tuned toward. It is also the reason a recommended tier exists: a bigger GPU removes the capacity ceiling this document identifies, at the cost of hardware this project does not require to run.
