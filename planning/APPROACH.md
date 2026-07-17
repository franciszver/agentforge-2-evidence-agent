# Portfolio Project Delivery Playbook

The repeatable process for turning each project brief in this folder into hiring-grade portfolio material. Written for the AgentForge sequence first (Phases 1–3), reusable for the standalone briefs after.

## The Goal

Every completed project produces exactly four things:

1. **A public GitHub repo** — production-quality, secure, runnable by a stranger in one command.
2. **A demo video (3–5 min)** — you narrate, following the script.
3. **A demo script** (`DEMO_SCRIPT.md`) — features, how it was built, architecture highlights, and "how this would run in production" talking points.
4. **Interview mastery** — you can defend every architectural decision without notes (`INTERVIEW_PREP.md` + a mock-interview session).

The standard, straight from the Phase 1 brief: *"The deliverable that matters is not the one that looks most impressive in a demo. It's the one you could defend in front of a hospital CTO."*

## Operating Decisions (already made — don't re-litigate per project)

- **Deployment:** Local-only via Docker Compose. Every project's ARCHITECTURE.md and DEMO_SCRIPT.md must include a credible **"Path to Production"** section (what changes for a real deployment: hosting, TLS, secrets management, managed DB, scaling, monitoring). You talk to this in the video instead of showing a URL.
- **Data:** Synthetic/demo data only. Never anything resembling real PHI or personal data. This is checked before the repo goes public.
- **Workflow:** Claude builds at full speed and keeps a decision log as it goes; study materials are generated at the end of each phase, followed by a mock interview before you record the video.
- **AgentForge repo structure: three standalone repos, one per phase, each with a unique URL** (submissions require unique URLs per project). None of them are GitHub forks — forks wear a "forked from" badge, get filtered from profile views, and commits to forks don't count toward the contribution graph. Instead:
  - `agentforge-1-clinical-copilot` (Phase 1): full-history **duplicate** of `Gauntlet-HQ/openemr-base-clean` (clone it, rename that remote to `upstream`, push to a fresh repo). Credit the Gauntlet base in the README.
  - `agentforge-2-evidence-agent` (Phase 2): full-history duplicate of repo 1 at its final Phase 1 state. README opens with "continues agentforge-1-clinical-copilot at v1.0." Repo 1 is then frozen as-completed; Phase 1 bugs found later get fixed here only.
  - `agentforge-3-redteam` (Phase 3): fresh independent repo (the brief requires this), targeting the Phase 1–2 Co-Pilot.
  - All three READMEs cross-link the series. Repos start **private** and flip to public only after the Stage 4 checklist passes.

## The Pipeline (per project / per phase)

Run these stages in order. Each has a verification gate — don't move on until it passes.

### Stage 1 — Foundation
- Clone/fork the base codebase into its own directory under `C:\Users\franc\Projects\` (NOT inside this master folder).
- Get it running locally with sample data; document setup as you go (this becomes the README setup guide).
- **Gate:** app runs from a fresh clone with one documented command.

### Stage 2 — Brief-Mandated Docs
- Produce whatever documents the brief hard-gates (Phase 1: `AUDIT.md`, `USERS.md`, `ARCHITECTURE.md` — each starting with a ~500-word summary).
- These aren't busywork: they're the source material for both the demo script and interview prep, and reviewers on the Alumni Roster will read them.
- **Gate:** every planned agent capability traces to a use case; every use case traces to the audit.

### Stage 3 — Build
- Implement per the brief's requirements (for AgentForge: agent + tools, authorization, verification layer, observability with correlation IDs, eval suite, health/ready endpoints, strict schemas, API collection, dashboards, load-test baselines).
- **Decision log discipline:** every non-obvious choice (framework, verification design, tradeoff) gets a 3-line entry in `docs/DECISIONS.md` *at the moment it's made* — decision, alternatives rejected, why. This is cheap during the build and impossible to reconstruct honestly afterward. It feeds INTERVIEW_PREP.md.
- **Gate:** eval suite passes; evals cover boundaries/invariants/regressions, not just happy paths.

### Stage 4 — Hardening & Repo Polish (the "respectable public repo" checklist)
- [ ] No secrets anywhere in git history (scan with gitleaks or equivalent; if history is dirty, fix before publishing, not after)
- [ ] `.env.example` with every required variable documented; real `.env` gitignored
- [ ] Synthetic data only — grep for anything that looks like real PII/PHI
- [ ] `/security-review` and `/simplify` passes; fix findings
- [ ] Tests + lint running in CI (GitHub Actions) with a status badge
- [ ] One-command setup (`docker compose up`) verified from a fresh clone
- [ ] README: what/why, architecture diagram, screenshots or GIF, setup guide, eval results, cost analysis, Path to Production section
- [ ] LICENSE file (check the base repo's license — OpenEMR is GPL, the fork must stay compatible)
- [ ] Runnable API collection (Bruno/Postman) committed
- [ ] Repo description, topics/tags, and pinned on your GitHub profile
- [ ] Flip the repo from private to public — only after every item above passes
- **Gate:** a stranger with Docker can go from `git clone` to working demo without asking you anything.

### Stage 5 — Demo Assets
- Write `DEMO_SCRIPT.md` using the template below.
- Prepare demo state: seeded data that makes the demo compelling (a patient whose story shows off retrieval, verification, and a graceful failure).
- Do a full timed dry run of the script against the running app before recording.
- **Gate:** dry run fits in 5 minutes and every claim in the script is demonstrably true on screen.

### Stage 6 — Learning & Interview Prep
- Generate `INTERVIEW_PREP.md`: Q&A format, seeded from the brief's own Interview Preparation section (audit findings, verification design, trust boundaries, eval insights, scaling to 500-bed hospital / 300 concurrent users, scariest failure mode) plus every entry in `docs/DECISIONS.md`.
- Include a plain-language architecture walkthrough (the "whiteboard explanation" you'd give in an interview) and a glossary of every technology used and why it was chosen over the alternative.
- **Mock interview:** run a Claude session that interrogates you using INTERVIEW_PREP.md — it asks, you answer out loud/typed, it probes gaps. Repeat until you're not reaching for notes.
- **Gate:** you can answer "why did you design the verification layer that way?" and "what breaks first at scale?" cold. Record the video only after this gate.

## Demo Script Template (3–5 min)

| Time | Segment | Content |
|------|---------|---------|
| 0:00–0:30 | Hook | The problem in human terms (e.g., "A physician has 90 seconds between rooms"). What you built in one sentence. |
| 0:30–2:30 | Product demo | 2–3 core flows on the real running app. Include one **deliberate failure case** handled gracefully — this impresses more than a third happy path. |
| 2:30–3:30 | Under the hood | Architecture diagram, the verification layer catching something, the observability dashboard tracing a request end-to-end, evals running. |
| 3:30–4:30 | Engineering story | How it was built: the audit-first approach, the hardest problem and how you solved it, one tradeoff you made deliberately. **Path to Production:** how local differs from a real deployment and exactly what you'd change (secrets manager, managed DB, TLS/BAA-compliant LLM endpoint, horizontal scaling, alerting). |
| 4:30–5:00 | Close | Eval results / metrics, what you'd build next, link to repo. |

Script format: two columns — **what's on screen** / **what you say**, written in your speaking voice, ~140 words per minute.

## AgentForge Sequencing

1. **Phase 1 — Clinical Co-Pilot Foundation** (`prds/1_AgentForge_ClinicalCoPilot_Foundation.html`): repo `agentforge-1-clinical-copilot`, duplicated from `github.com/Gauntlet-HQ/openemr-base-clean`, full pipeline above. This phase carries the most fixed requirements (audit hard gate, correlation IDs, schemas, dashboards, load tests) — budget the most time here. Good architecture here compounds; debt costs double in Phases 2–3.
2. **Phase 2 — Multimodal Evidence Agent & Document RAG** (`prds/2_...html`): repo `agentforge-2-evidence-agent`, duplicated with full history from repo 1's final state. New scope: document ingestion, vision extraction, hybrid RAG, supervisor + 2 workers, 50-case golden eval set as a CI gate, citation contracts with PDF overlays. Re-run Stages 2–6; new demo video + script.
3. **Phase 3 — Adversarial Security & Red Team Platform** (`prds/3_...html`): repo `agentforge-3-redteam`, fresh and independent (per the brief), whose attack target is the Phase 2 Co-Pilot. Threat model with OWASP Top 10 + OWASP LLM Top 10 mapping, red team / judge / orchestrator / documentation agents, exploit database, regression harness, cost-tiered model selection, minimum three vulnerability reports. Re-run Stages 2–6; new demo video + script.

Each phase completion = a frozen repo with its own unique URL, README, demo video, and INTERVIEW_PREP.md. Kickoff prompts for Phases 2 and 3 get generated at the end of the preceding phase (see `INITIAL_PROMPT.md`).

## Known Risk of the "Build Fast, Docs After" Choice

The failure mode: a finished repo you can't defend, which is worse for hiring than a smaller repo you understand completely. Mitigations built into this playbook: the decision log is written *during* the build (Stage 3), and the mock-interview gate (Stage 6) blocks video recording until you pass it. Don't skip Stage 6 — it is the difference between a portfolio and a liability.
