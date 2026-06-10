# atlas-agent-runtime

Hand-rolled thin agent loop for Atlas. No LangGraph — a purpose-built Python 3.12 asyncio engine that executes YAML-defined agents with hard caps on iterations, tokens, and wall-time. (Pydantic AI is the sanctioned fallback if the hand-rolled cost isn't affordable — see [ADR-006](../atlas-docs/02-tech-stack-and-adrs.md).) Runs are triggered over a thin **FastAPI** surface ([ADR-020](../atlas-docs/02-tech-stack-and-adrs.md)).

## Purpose

Runs structured agents (e.g. RegDoc Q&A) against the Atlas Gateway LLM API and a set of MCP tool servers. Each agent is declared as a YAML file; the runtime validates the spec, enforces resource caps, whitelists tool calls, sanitizes tool results before they re-enter context, and persists every run and step to PostgreSQL with full OpenTelemetry tracing.

---

## Agent Definition Schema (YAML)

```yaml
# Example: regdoc-qa.yaml
agent_name: regdoc-qa
agent_version: "1.0.0"

system_prompt_ref: prompts/regdoc-qa.txt   # path to system prompt file
model_alias: atlas-default                  # resolved by the Gateway

tool_whitelist:
  - doc_search
  - verify_citation

max_iterations: 10       # hard cap — breach raises CapBreachError
token_budget: 32000      # hard cap — breach raises CapBreachError
timeout_s: 120           # wall-time hard cap — breach raises CapBreachError
```

### Schema fields

| Field | Type | Description |
|---|---|---|
| `agent_name` | `str` | Unique name for the agent |
| `agent_version` | `str` | Semver string |
| `system_prompt_ref` | `str` | Relative path to the system prompt file |
| `model_alias` | `str` | Logical model name resolved by the Gateway |
| `tool_whitelist` | `list[str]` | Exact tool names allowed; any other call is rejected |
| `max_iterations` | `int` | Maximum LLM→tool cycles before `CapBreachError` |
| `token_budget` | `int` | Cumulative token cap across all LLM calls |
| `timeout_s` | `int` | Wall-time limit in seconds for the full run |

---

## Hard Caps

The runtime enforces three independent hard caps. A breach on **any** one immediately raises `CapBreachError` with an explicit message identifying the cap type — no silent failures.

| Cap | Field | Error message |
|---|---|---|
| Iteration | `max_iterations` | `CapBreachError: max_iterations (10) exceeded` |
| Token | `token_budget` | `CapBreachError: token_budget (32000) exceeded` |
| Wall-time | `timeout_s` | `CapBreachError: timeout_s (120s) exceeded` |

Runs that hit a cap are persisted with `status = "capped"`.

---

## Module Map (`app/`)

```
app/
├── api/               # FastAPI trigger surface: POST /v1/agent/runs, GET /v1/agent/runs/{id}, /healthz (ADR-020, AGT-16)
├── main.py            # FastAPI composition root (uvicorn app.main:app --port 8000)
├── config.py          # env-driven Settings (ATLAS_ prefix); persistence config-gated on ATLAS_DATABASE_URL
├── loop/              # AgentRunner — main asyncio run loop
├── agentspec/         # Pydantic model for YAML agent definitions (AgentSpec)
├── tools/
│   ├── registry/      # ToolRegistry — whitelist enforcement; rejects unknown tools
│   └── sanitize/      # ToolSanitizer — injection screen before result re-enters context
├── persistence/       # SQLAlchemy models + DAL + session factory for agent_runs / agent_steps
├── telemetry/         # OTel span instrumentation (gen_ai.operation.name, gen_ai.agent.name)
└── gateway_client/    # HttpGatewayClient — HTTP client to the Atlas Gateway (/v1/chat/completions)
```

> **MCP tool execution** is intentionally outside the runner (tool calls are
> whitelisted + sanitized but executed by the caller). A dedicated MCP client
> (`app/mcp_client/`) lands when the agent loop is wired to execute tools
> in-loop; the current trigger surface runs the loop against the gateway and
> persists runs/steps.

### Trigger surface (AGT-16)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/agent/runs` | Run an agent (`{agent_name, user_message}`); returns final content + run id |
| `GET` | `/v1/agent/runs/{id}` | Fetch a persisted run's status + steps (503 if no `ATLAS_DATABASE_URL`) |
| `GET` | `/healthz` | Liveness |

Agent specs are resolved from `ATLAS_AGENTS_DIR` (default `agents/`) as
`<agent_name>.yaml`. With no provider keys the gateway serves `model=mock`, so
`agents/regdoc-qa.yaml` runs end-to-end offline.

---

## External Dependencies

| Dependency | Role |
|---|---|
| **Atlas Gateway** | LLM API — OpenAI-compatible `/v1/chat/completions` |
| **mcp-doc-search** | `doc_search(query, k)` — hybrid BM25 (Elasticsearch) + vector (Qdrant) search |
| **mcp-citations** | `verify_citation(source_id, claim)` — validates claims against source snippets |
| **PostgreSQL** | Persistent store for `agent_runs` and `agent_steps` |
| **OTel Collector** | Receives spans via OTLP |

### Citation Enforcement

After the agent composes an answer, a post-guardrail step calls `verify_citation` for every claim. Answers that reference a `source_id` not returned by a prior `doc_search` call are **rejected** — the run fails with an explicit citation error.

---

## Database Schema

### `agent_runs`

| Column | Type | Notes |
|---|---|---|
| `id` | UUID PK | |
| `agent_name` | text | |
| `agent_version` | text | |
| `status` | text | `created`, `running`, `completed`, `failed`, `capped` |
| `token_budget` | int | from AgentSpec |
| `tokens_used` | int | accumulated |
| `started_at` | timestamptz | |
| `ended_at` | timestamptz | nullable |

### `agent_steps`

| Column | Type | Notes |
|---|---|---|
| `id` | UUID PK | |
| `agent_run_id` | UUID FK → `agent_runs.id` | |
| `idx` | int | step index within run |
| `type` | text | `llm_call` or `tool_call` |
| `payload` | jsonb | request/response payload |
| `tokens` | int | tokens for this step |
| `latency_ms` | int | |

---

## OTel Instrumentation

Spans follow the GenAI semantic conventions:

| Attribute | Values |
|---|---|
| `gen_ai.operation.name` | `invoke_agent`, `execute_tool` |
| `gen_ai.agent.name` | agent name from `AgentSpec` |

---

## Python Version & Dependencies

- **Python**: 3.12
- See `requirements.txt` for pinned versions. Key dependencies:
  - `fastapi` + `uvicorn` (run-trigger surface — ADR-020)
  - `anthropic` / gateway HTTP client
  - `mcp` (official Python MCP SDK)
  - `pydantic>=2`
  - `sqlalchemy>=2`
  - `asyncpg`
  - `opentelemetry-sdk`
  - `opentelemetry-exporter-otlp`
  - `pyyaml`

No secrets in code — all credentials via environment variables.

---

## Diagrams

- [C4-L3 Component Diagram](docs/diagrams/component-c4.md)
- [Agent Loop Class Diagram](docs/diagrams/agent-loop-class.puml)
- [Agent Run State Machine](docs/diagrams/agent-run-state.md)
- [RegDoc Agent Sequence](docs/diagrams/seq-agent-rag.md)
- [Loop Control / Hard-Cap Flow](docs/diagrams/flow-loop-control.md)

---

## Related

- [atlas-docs](../atlas-docs) — system-wide documentation
- Atlas Gateway — LLM routing and model alias resolution
- mcp-doc-search — document search MCP server
- mcp-citations — citation verification MCP server
