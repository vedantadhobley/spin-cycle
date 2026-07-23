# spin-cycle — Agent Context

Recurring / scheduled task project. Temporal-based worker + orchestration. Dev + prod stacks currently running (8 containers). Active branch varies — check `git status` before making assumptions.

## Cross-cutting context

Workspace-wide rules, node topology, and cross-project decisions live in [`~/workspace/vedanta-dhobley/`](../../vedanta-dhobley/). Every agent session reads its global `AGENTS.md` automatically via symlinks (`~/.claude/CLAUDE.md`, `~/.codex/AGENTS.md`, `~/.gemini/GEMINI.md`); this pointer exists so anyone browsing the repo sees the pattern.

- [`AGENTS.md`](../../vedanta-dhobley/AGENTS.md) — operating model, commit conventions, Docker-first policy, host-port scheme, `mem_limit` rules, tailnet FQDN rule, privacy preferences
- [`docs/topology.md`](../../vedanta-dhobley/docs/topology.md) — aerial view of nodes, services, routing, messaging, roadmap
- [`docs/decisions.md`](../../vedanta-dhobley/docs/decisions.md) — timestamped rationale for locked-in choices (joi model swap affects this project's LLM endpoints)
- [`docs/plans/`](../../vedanta-dhobley/docs/plans/) — active time-bounded plans

**Where things belong:** if a decision in this project turns out to be cross-project, raise it in dhobley — do not duplicate it here.

## Scope

Placeholder — this AGENTS.md is a hygiene stub written from outside the project. Fill in from a spin-cycle session with an agent that knows the code:

- Stack detail (Temporal shape, DBs, external dependencies)
- Prod/dev container names + routes through Caddy
- Where LLM calls target (should be env-driven per the workspace pattern; `LLM_ENDPOINT` env → `llama-small.joi` today, swappable to nexus dispatch later)
- Any project-specific conventions

## Where to look first

- [dhobley's AGENTS.md](../../vedanta-dhobley/AGENTS.md) — workspace-wide rules
- `docs/` — existing project docs (if present)
- `docker-compose.yml` / `docker-compose.dev.yml` — the running stacks
