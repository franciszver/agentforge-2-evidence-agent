# Clinical Co-Pilot — Developer's Guide

- **Status:** living document. This is the orientation hub for anyone (human or AI agent) joining the project — it tells you where things are and how work flows, then links to the canonical sources. It deliberately does **not** duplicate rule text: each rule lives in exactly one place.

## What this project is

An AI Clinical Co-Pilot embedded in OpenEMR: a physician asks questions about the patient whose chart is open ("what changed since her last visit?", "does anything she's taking conflict with ibuprofen?") and gets answers where **every factual claim is deterministically verified against the record and cited**. Inference is local-only (Ollama, small model) — PHI never leaves the machine. The full design rationale is in `docs/IMPLEMENTATION_PLAN.md`; the architecture summary is in `ARCHITECTURE.md`.

## Where things are

| Area | Location |
|---|---|
| Agent service (FastAPI, verification layer, tools) | `services/copilot-agent/` |
| OpenEMR UI module (panel, render events, token broker) | `interface/modules/custom_modules/oe-module-clinical-copilot/` |
| Eval suite (YAML cases + harness) | `evals/` |
| Dev stack compose overlay | `docker/development-easy/docker-compose.copilot.yml` |
| CI | `.github/workflows/copilot-ci.yml` |
| Everything else | upstream OpenEMR — treat as the platform, not the project |

## The canonical sources (read in this order)

1. **`CLAUDE.md` (root) — "Clinical Co-Pilot Development Operations."** The execution model: the two hard invariants (nothing reaches `main` without a branch + PR; every PR links a board issue), agent roles, and the per-task pipeline. If you're an AI agent, this is your contract.
2. **`docs/TEST_PLAN.md`** — how everything is tested: strict TDD and the red-first artifact per code type, the three pre-push gates, the PR Definition of Done checklist, coverage policy, eval authoring, fixtures, CI record/replay. Living document.
3. **The GitHub Project board** (URL in `README.md`) — the live operational plan. All work items, their acceptance criteria, and their status. If it isn't on the board, it isn't planned; if you're about to do it anyway, put it on the board first.
4. **`docs/IMPLEMENTATION_PLAN.md`** — the frozen founding plan: phases, architecture, decisions. Historical context; the board supersedes it for day-to-day truth.
5. **`CONTRIBUTING.md` + root `CLAUDE.md` (lower sections)** — OpenEMR dev-stack setup, `openemr-cmd`, testing commands, worktrees.

## How a change ships — worked example

Say you're implementing the allergy cross-check (a Phase 3 board issue):

1. **Claim the issue** on the board; move it to *In Progress*. Read its acceptance criteria — they came from the plan's verify gates.
2. **Branch:** `feat/p3-allergy-crosscheck` off `main`.
3. **Red first:** write the failing tests — match/no-match/case-variant/compound-name — and commit them failing. This commit is deliberately visible in the PR history.
4. **Green:** implement until the tests pass. **Refactor** with tests green. Commit at each green boundary (conventional commits, `Assisted-by` trailer if an AI helped).
5. **Gates:** run `/simplify` → `/security-review` → `/code-review` on the full diff. Fix every finding; re-run the full suite green.
6. **PR:** push the branch, open a PR with `Closes #<issue>`. CI must pass. If your change touched UI, scenarios ran at desktop + 360×800 viewports; if it touched agent behavior, you ran a live eval locally and committed the updated recordings (see TEST_PLAN §9).
7. **Merge.** The issue auto-closes, the board updates, and `main` stays green.

Found a bug or a missing piece mid-task? Don't fix it on your branch as a stowaway and don't touch `main` — **create a board issue first**, then decide with the orchestrator whether it blocks the current PR or ships as its own.

## Running the project

- **Stack up:** see `CONTRIBUTING.md` quick start plus the Co-Pilot overlay: `docker compose -f docker-compose.yml -f docker-compose.copilot.yml up` from `docker/development-easy/`.
- **Tests:** commands and cadence in `docs/TEST_PLAN.md` §8.
- **App:** OpenEMR at the usual dev ports (`admin`/`pass` on demo data); Co-Pilot panel on the patient dashboard; standalone PWA at `/copilot`.
- **Phone access over Tailscale:** `scripts/tailscale-serve-copilot.sh up` maps the stack to tailnet HTTPS (`status`/`down` also available); see the script header for what it exposes and the one known limitation (OAuth consent redirect flow, off by default in dev).

## For AI agents specifically

Your session inherits the execution model from root `CLAUDE.md` automatically. The two things most often gotten wrong: (1) committing implementation before the failing test — the red commit must come first and be visible; (2) doing untracked side-work — every diff you produce must trace to a board issue via its PR. When in doubt, stop and check the board.
