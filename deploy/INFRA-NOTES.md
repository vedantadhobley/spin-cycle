# Deploy notes — what changes outside this repo

The application code is fully contained in this repo. The host-level
infrastructure changes needed to run spin-cycle on luv live in the
workspace's `~/workspace/proxy/` stack — they aren't tracked here because
they're cross-cutting, but they're listed here so the deploy is
reproducible from one place.

spin-cycle has **no public-facing component** (no Cloudflare tunnel
ingress); the api / temporal-ui / adminer are tailnet-only.

## 1. Caddyfile additions (luv)

Append to `~/workspace/proxy/Caddyfile`:

```caddy
# ─── spin-cycle (prod + dev) ───────────────────────────────────────────────
# api standardized to internal :3000 in both envs (was :3500 prod / :4500 dev).

http://spin-cycle-prod-api.{$BASE_DOMAIN}          { reverse_proxy spin-cycle-prod-api:3000 }
http://spin-cycle-prod-temporal-ui.{$BASE_DOMAIN}  { reverse_proxy spin-cycle-prod-temporal-ui:8080 }
http://spin-cycle-prod-adminer.{$BASE_DOMAIN}      { reverse_proxy spin-cycle-prod-adminer:8080 }

http://spin-cycle-dev-api.{$BASE_DOMAIN}           { reverse_proxy spin-cycle-dev-api:3000 }
http://spin-cycle-dev-temporal-ui.{$BASE_DOMAIN}   { reverse_proxy spin-cycle-dev-temporal-ui:8080 }
http://spin-cycle-dev-adminer.{$BASE_DOMAIN}       { reverse_proxy spin-cycle-dev-adminer:8080 }
```

Then: `docker compose -f ~/workspace/proxy/docker-compose.yml restart caddy`.

## 2. Cross-project network dependency

spin-cycle's postgres is reached from `vedanta-systems-prod-api` over the
shared `luv-prod` docker network. That network must exist:

```bash
docker network create luv-prod
docker network create luv-dev
```

(One-time on a fresh node; idempotent if already created.)

## 3. LLM dependency

The api and worker reach the local LLM endpoints on the joi node via
tailnet split-DNS:

- `LLAMA_URL=http://llama-large.joi`        (chat, Qwen3.5-122B-A10B)
- `LLAMA_EMBED_URL=http://llama-embed.joi`  (embeddings, Qwen3-Embedding-8B)

These resolve via the Tailscale split-DNS rule for the `joi` domain
(configured once in the Tailscale admin console). No fallback — if joi
is offline, LLM-dependent activities fail. See `~/workspace/proxy/README.md`
for the split-DNS setup.

## 4. Bring up

```bash
cd ~/workspace/dev/spin-cycle
cp .env.example .env
$EDITOR .env                                  # set passwords + API keys
docker compose -f docker-compose.yml up -d --build         # prod
# or
docker compose -f docker-compose.dev.yml up -d --build     # dev
```

## 5. Verify

```bash
curl -sI http://spin-cycle-prod-api.luv/health
curl -sI http://spin-cycle-prod-temporal-ui.luv/
curl -sI http://spin-cycle-prod-adminer.luv/
```
