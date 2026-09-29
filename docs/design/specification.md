# CogMem — Production-Grade Build Specification

**Product Requirements · High-Level Design · Technical Architecture · Implementation Roadmap**

## 1. Product Vision

Build CogMem, an open-source, production-grade persistent memory platform that allows AI agents to remember, retrieve, update, and share information across sessions, projects, and coding tools.

**Core value proposition:** Persistent, searchable memory for AI agents, accessible through MCP and a hosted cloud service.

An agent should be able to remember a repository convention in Claude Code and retrieve it in Codex or Cursor in a later session, with appropriate permissions.

### Target users

- Individual developers using Claude Code, Codex, Cursor, VS Code agents, and other MCP clients.
- AI engineers building custom agents with LangGraph, LangChain, or the OpenAI SDK.
- Teams that need shared project and organizational memory.
- Researchers evaluating whether persistent memory improves agent performance.

### Product deliverables

| Component | Purpose |
|---|---|
| CogMem Core | Open-source memory engine |
| CogMem MCP Server | Standard interface for compatible AI agents |
| CogMem Cloud | Hosted, multi-tenant memory service |
| CogMem SDK | Python SDK and programmatic access |
| CogMem CLI | Local setup, configuration, and diagnostics |
| CogMem Console | Web dashboard for managing memory, projects, and usage |

## 2. High-Level Design (HLD)

### System architecture

```text
                  AI AGENT CLIENTS
       Claude Code | Codex | Cursor | Custom Agents
                         |
                         | MCP Protocol
                         v
              +------------------------+
              |   CogMem MCP Gateway   |
              | Auth, Rate Limits, ACL |
              +------------------------+
                         |
              +------------------------+
              |     MCP Server         |
              |    FastMCP / Python    |
              +------------------------+
                         |
              +------------------------+
              |    Memory Service      |
              |------------------------|
              | Memory Manager         |
              | Retrieval Engine       |
              | Context Builder        |
              | Conflict Resolution    |
              | Memory Lifecycle       |
              +------------------------+
                  |               |
                  v               v
         +----------------+  +----------------+
         | PostgreSQL     |  | Background     |
         | pgvector       |  | Workers        |
         |                |  |                |
         | Memories       |  | Embeddings     |
         | Relationships  |  | Extraction     |
         | Metadata       |  | Consolidation  |
         +----------------+  +----------------+
                  |
                  v
         +-----------------------+
         | REST API / Admin API  |
         +-----------------------+
                  |
          +-------+--------+
          |                |
          v                v
    CogMem Console     Python SDK / CLI
```

### Architectural principles

1. Keep the memory engine independent of MCP, so custom applications can use it directly.
2. Separate the open-source core from cloud-specific authentication, billing, and management.
3. Enforce tenant and memory-space isolation at every access point.
4. Use asynchronous processing for expensive extraction, embedding, and consolidation.
5. Make model providers, embedding providers, storage, and job queues pluggable.
6. Never require a paid LLM API for the basic local or self-hosted experience.

## 3. Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12+ |
| MCP server | FastMCP, compatible with the supported MCP SDK |
| REST API | FastAPI |
| Validation | Pydantic v2 |
| ORM | SQLAlchemy 2.x |
| Migrations | Alembic |
| Hosted database | PostgreSQL + pgvector |
| Local database | SQLite |
| Cache / rate limiting | Redis, introduced when needed |
| Background processing | Lightweight worker initially; Redis-backed queue later |
| Embeddings | Sentence Transformers / configurable local models |
| Frontend | React + TypeScript |
| Testing | pytest, async tests, integration tests |
| Deployment | Docker, managed PostgreSQL, cloud container hosting |
| Observability | OpenTelemetry, structured logging, Prometheus-compatible metrics |

Avoid introducing Kafka, Kubernetes, Neo4j, or multiple microservices in the initial MVP. Begin with a modular monolith and extract components only when actual workload or deployment requirements justify it.

## 4. Memory Architecture

CogMem must support five memory layers.

| Memory layer | Stores | Example |
|---|---|---|
| Semantic | Stable facts and preferences | User prefers Python for ML projects |
| Episodic | Events and interactions | Agent fixed a database migration on September 20 |
| Temporal | Facts with validity periods | Project deadline changed to October 15 |
| Graph | Entities and relationships | Project A uses PostgreSQL and FastAPI |
| Reflective | Consolidated patterns and insights | User consistently prefers minimal dependencies |

Each memory should have a common representation.

```python
class MemoryRecord(BaseModel):
    id: UUID
    tenant_id: UUID
    space_id: UUID

    content: str
    memory_type: str

    source: str | None = None
    entity_ids: list[UUID] = []

    created_at: datetime
    updated_at: datetime
    valid_from: datetime | None = None
    valid_until: datetime | None = None

    importance: float = 0.5
    confidence: float = 1.0

    metadata: dict = {}
```

Add appropriate validators, immutable identifiers, timezone-aware timestamps, and database constraints in the implementation.

### Memory lifecycle

```text
Agent submits information
          |
          v
   Validate + Authorize
          |
          v
    Normalize content
          |
          v
    Deduplicate / Merge
          |
          v
    Store raw memory
          |
          v
 Extract entities / Embed
          |
          v
     Index and retrieve
          |
          v
  Update / Expire / Forget
```

The memory write path should not require an LLM call. Basic storage must work with deterministic logic and local embeddings. LLM-based extraction and reflection should be optional, asynchronous capabilities.

## 5. Memory Spaces and Multi-Tenancy

Memory must be scoped to prevent unrelated agents, projects, or users from seeing each other's information.

### Memory-space types

- `personal`: Individual user preferences and history.
- `project`: Shared information for a particular project.
- `repository`: Codebase-specific conventions, architecture, and decisions.
- `organization`: Approved shared organizational knowledge.
- `agent`: Agent-specific state and working memory.

Example:

```text
Tenant
 ├── Personal Space
 ├── Project: CogMem
 │    ├── Repository: cogmem-core
 │    └── Repository: cogmem-dashboard
 └── Organization Space
```

Every memory operation must specify or resolve an authorized memory space.

Use PostgreSQL Row-Level Security where appropriate, combined with application-level authorization. Never trust a client-supplied tenant or space identifier without verifying access.

## 6. MCP Interface

MCP is the primary integration mechanism for third-party agents.

Implement the following tools with Pydantic input/output schemas, clear descriptions, and predictable error handling.

### Core MCP tools

| Tool | Description |
|---|---|
| `remember` | Store a fact, decision, event, or preference |
| `recall` | Retrieve relevant memories using a natural-language query |
| `search_memory` | Search memories with filters and ranking controls |
| `get_context` | Build a token-budgeted context for the current task |
| `update_memory` | Correct, merge, or supersede an existing memory |
| `forget` | Delete or invalidate selected memories |
| `get_memory` | Retrieve a memory by ID |

### Advanced tools

| Tool | Description |
|---|---|
| `get_relationships` | Traverse entity relationships |
| `get_history` | Retrieve historical versions or temporal changes |
| `create_memory_space` | Create an authorized memory space |
| `list_memory_spaces` | List accessible spaces |
| `memory_stats` | Return usage and memory statistics |

Keep the initial tool surface small. Implement the seven core operations first, then add advanced tools once their semantics and authorization rules are tested.

### Example MCP interface

```python
@mcp.tool()
async def remember(
    content: str,
    memory_type: str = "semantic",
    space_id: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """Store a persistent memory in an authorized memory space."""
    ...
```

```python
@mcp.tool()
async def recall(
    query: str,
    space_id: str | None = None,
    limit: int = 5,
) -> dict:
    """Retrieve relevant memories for the current task."""
    ...
```

The actual implementation must derive the authenticated tenant and user from the server-side security context, not from untrusted tool arguments.

## 7. Retrieval Engine

Retrieval quality is the core of CogMem. Storing information is insufficient if the system cannot retrieve the right memory at the right time.

### Retrieval pipeline

```text
Incoming query
      |
      v
Query normalization
      |
      v
Space and permission filtering
      |
      v
+-------------------------------+
| Parallel retrieval            |
|                               |
| Vector similarity             |
| Keyword / full-text search    |
| Temporal filtering            |
| Graph relationship lookup     |
+-------------------------------+
      |
      v
Candidate fusion and ranking
      |
      v
Deduplication and conflict checks
      |
      v
Token-budgeted context assembly
      |
      v
Return memories + provenance
```

### Ranking signals

The ranking function should support configurable weights for:

- Semantic similarity.
- Keyword relevance.
- Recency and temporal validity.
- Importance and confidence.
- Memory type and source.
- Relationship or graph relevance.

Start with a transparent weighted ranking function. Introduce learned rerankers only after benchmark results demonstrate a measurable improvement.

### Context builder

`get_context` should return a compact, structured context that an agent can inject into its prompt.

Requirements:

- Accept a task description or query.
- Accept a token budget.
- Return the most relevant authorized memories.
- Include source IDs and timestamps.
- Exclude expired or superseded memories by default.
- Handle contradictions explicitly.
- Return an empty result cleanly when no relevant memory exists.

Do not inject the entire memory database into the agent's context window.

## 8. Database Design

Use PostgreSQL and pgvector for hosted deployments. SQLite should support the local self-hosted edition.

### Core tables

| Table | Purpose |
|---|---|
| `users` | User identity and account metadata |
| `organizations` | Organization or tenant records |
| `memberships` | User roles and organization membership |
| `api_keys` | Hashed API keys and their permissions |
| `memory_spaces` | Personal, project, repository, and organization spaces |
| `memories` | Memory content, metadata, timestamps, embeddings |
| `entities` | Extracted entities such as projects, people, and technologies |
| `relationships` | Edges between entities and memories |
| `memory_sources` | Source agent, conversation, repository, or event |
| `memory_versions` | Optional history for memory edits and supersession |
| `usage_events` | Usage accounting and quotas |

Use UUID primary keys, indexes for tenant/space access, vector indexes for semantic retrieval, and database migrations for every schema change.

Avoid storing secrets, raw credentials, or unnecessary sensitive user data in memory content.

## 9. Cloud API and Authentication

### REST API

Expose management and integration endpoints independently of the MCP transport.

```text
POST   /v1/memories
GET    /v1/memories
GET    /v1/memories/{id}
PATCH  /v1/memories/{id}
DELETE /v1/memories/{id}

POST   /v1/recall
POST   /v1/context

POST   /v1/spaces
GET    /v1/spaces
GET    /v1/spaces/{id}

GET    /v1/usage
GET    /v1/health
```

The MCP server and REST API must both call the same memory service. Do not duplicate business logic across transports.

### Authentication

For the first hosted MVP:

- API keys with cryptographically secure generation.
- Store only hashed API keys.
- Support key revocation and rotation.
- Scope keys to users, organizations, and permitted memory spaces.
- Enforce request-size limits, rate limits, and usage quotas.

For public hosted MCP integrations, implement the appropriate MCP authorization flow, including OAuth where supported by the target clients. Do not assume that every coding agent uses an identical OAuth configuration.

## 10. Provider Abstractions

Avoid tightly coupling CogMem to any single model provider.

Define interfaces such as:

```python
class EmbeddingProvider(Protocol):
    async def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        ...


class MemoryRepository(Protocol):
    async def save(self, memory: MemoryRecord):
        ...

    async def get(self, memory_id: UUID):
        ...

    async def delete(self, memory_id: UUID):
        ...


class MemoryRetriever(Protocol):
    async def search(
        self,
        query: str,
        space_id: UUID,
        limit: int = 5,
    ):
        ...
```

Provide implementations for local embeddings and the initial database backends. Keep cloud-specific and external LLM integrations optional.

The basic CogMem experience must remain usable without OpenAI, Anthropic, or another paid model API.

## 11. Background Processing

Implement asynchronous workers for expensive operations.

| Job | Purpose |
|---|---|
| Embedding | Generate vectors for new or updated memories |
| Extraction | Extract entities and structured facts |
| Deduplication | Identify redundant memories |
| Consolidation | Merge related memories and summarize episodes |
| Reflection | Derive higher-level patterns |
| Expiration | Process expired or invalid memories |
| Maintenance | Rebuild indexes and clean up stale records |

Requirements:

- Idempotent jobs.
- Retries with exponential backoff.
- Dead-letter handling or a persistent failed-job state.
- Job status and observability.
- No silent loss of memory writes.

Start with a database-backed or lightweight Redis-backed worker. Add a more sophisticated queue only when workload requires it.

## 12. Developer Experience

The first-time integration must take minutes, not hours.

### Local installation

Provide a simple installation and configuration flow.

```bash
pip install cogmem
```

Provide commands conceptually equivalent to:

```bash
cogmem init
cogmem serve
cogmem doctor
cogmem config
```

The CLI must support local storage, configuration validation, MCP server startup, and diagnostics.

### Coding-agent integrations

Document working configuration examples for:

- Claude Code.
- Codex.
- Cursor.
- VS Code agents supporting MCP.
- Custom Python MCP clients.

Each integration guide should include:

1. Installation.
2. Server configuration.
3. Authentication for hosted mode.
4. A test that stores a memory.
5. A separate session or agent that retrieves the memory.

Do not advertise an integration as supported until it has been tested against the actual client configuration.

## 13. CogMem Console

Build a minimal web dashboard after the core MCP server and hosted API work.

### Required screens

| Screen | Functionality |
|---|---|
| Dashboard | Memory count, usage, active spaces |
| Memory Explorer | Search, inspect, edit, and delete memories |
| Memory Spaces | Create and manage personal/project spaces |
| Integrations | Create API keys and configure MCP clients |
| Usage | Requests, storage, quotas, latency |
| Organization | Members, roles, shared spaces |

The dashboard must enforce the same authorization policies as the MCP and REST interfaces.

## 14. Security, Privacy, and Reliability

Security is a core product requirement, not a later optimization.

Implement:

- Strict tenant isolation and space-level access control.
- Secure API-key generation, hashing, rotation, and revocation.
- Encryption in transit and encrypted managed database storage.
- Input validation, request-size limits, and rate limiting.
- Protection against prompt injection in retrieved memory content.
- Memory deletion and documented retention policies.
- No cross-tenant retrieval, embedding search, or graph traversal.
- No use of customer memories for model training without explicit consent.
- Audit logs for sensitive operations.
- Backup and restoration procedures.

Retrieved memory is untrusted data. Agents must not treat stored instructions as higher-priority system instructions. The retrieval layer should return memory content with provenance and clear data boundaries.

For production, define measurable service-level objectives for availability, latency, and recovery time before promising a public SLA.

## 15. Evaluation and Benchmarking

CogMem must demonstrate measurable value over agents without persistent memory.

Do not claim that CogMem improves accuracy, reduces hallucinations, or improves agent performance until experiments support those claims.

### Benchmark setup

Compare:

1. Baseline agent without persistent memory.
2. Agent with CogMem.
3. Optional full-history baseline, where the complete available conversation is supplied within a defined context budget.

Use the same tasks, model, prompt, and inference settings wherever possible.

### Evaluation categories

| Category | Example task |
|---|---|
| Factual recall | Recall a preference stored in an earlier session |
| Temporal reasoning | Identify the latest project deadline |
| Multi-hop retrieval | Connect a project with its database and deployment architecture |
| Personalization | Follow a previously recorded coding preference |
| Contradiction handling | Resolve a changed fact without returning the outdated value |
| Cross-agent memory | Store through one client and retrieve through another |
| Long-running tasks | Recall earlier decisions during a later project phase |

### Metrics

- Recall@k and Precision@k.
- Retrieval MRR and nDCG.
- Answer accuracy and task completion rate.
- Contradiction resolution accuracy.
- Memory write and retrieval latency (p50/p95).
- Tokens consumed by retrieved context.
- Cost per task, including embedding and optional LLM calls.
- Unauthorized retrieval rate, with a target of zero.

Run ablations for semantic-only, hybrid retrieval, graph-assisted retrieval, and temporal filtering. Publish benchmark datasets, configurations, and reproducible evaluation scripts.

## 16. Repository Structure

Use a modular, testable repository structure.

```text
cogmem/
├── pyproject.toml
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── SECURITY.md
├── Dockerfile
├── docker-compose.yml
├── .env.example
│
├── src/
│   └── cogmem/
│       ├── core/
│       │   ├── models.py
│       │   ├── config.py
│       │   └── exceptions.py
│       │
│       ├── memory/
│       │   ├── manager.py
│       │   ├── retrieval.py
│       │   ├── context.py
│       │   ├── lifecycle.py
│       │   └── conflict_resolution.py
│       │
│       ├── storage/
│       │   ├── interfaces.py
│       │   ├── sqlite.py
│       │   └── postgres.py
│       │
│       ├── embeddings/
│       ├── extraction/
│       ├── graph/
│       ├── temporal/
│       ├── reflection/
│       │
│       ├── mcp/
│       │   ├── server.py
│       │   └── tools.py
│       │
│       ├── api/
│       ├── auth/
│       ├── workers/
│       └── cli/
│
├── migrations/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── security/
│   └── evaluation/
│
├── benchmarks/
├── examples/
├── docs/
└── scripts/
```

Keep interfaces stable and avoid circular imports. Separate core memory logic from the MCP transport and cloud deployment code.

## 17. Implementation Roadmap

Build in the following sequence. Each phase must be tested and documented before moving on.

### Phase 1 — Local MCP Memory MVP

- Inspect and document the existing CogMem repository.
- Preserve and evaluate existing tools and memory-layer implementations.
- Implement local persistence with SQLite.
- Implement `remember`, `recall`, `search_memory`, `get_context`, `update_memory`, `forget`, and `get_memory`.
- Add basic semantic retrieval and local embeddings.
- Provide working MCP configuration examples.
- Add unit and integration tests.

**Exit criteria:** A developer can install CogMem locally, store information in one session, and retrieve it in another.

### Phase 2 — Production Memory Core

- Introduce the storage and embedding provider interfaces.
- Add PostgreSQL and pgvector.
- Implement Alembic migrations and indexing.
- Add tenant, user, and memory-space models.
- Implement hybrid retrieval, deduplication, provenance, and temporal validity.
- Add automated tests for deletion, updates, and conflicting memories.

### Phase 3 — CogMem Cloud

- Implement FastAPI management endpoints.
- Add API-key authentication, tenant isolation, quotas, and rate limits.
- Deploy MCP over a publicly accessible HTTPS endpoint.
- Configure managed PostgreSQL, backups, migrations, and monitoring.
- Add health checks, structured logs, and operational documentation.

### Phase 4 — Developer Adoption

- Build CLI and Python SDK.
- Write and test coding-agent integration guides.
- Publish a Docker-based self-hosted deployment.
- Add onboarding, configuration validation, and error diagnostics.
- Create a minimal hosted signup and API-key workflow.

### Phase 5 — Intelligent Memory

- Add background extraction and entity resolution.
- Implement graph relationship retrieval.
- Add temporal history and conflict resolution.
- Implement optional memory consolidation and reflection.
- Introduce configurable LLM providers and local fallbacks.

### Phase 6 — Evaluation and Optimization

- Create a reproducible benchmark dataset.
- Compare memoryless and CogMem-enabled agents.
- Measure retrieval quality, latency, token use, and cost.
- Run ablations and optimize ranking and context assembly.
- Publish results with limitations and reproducible configurations.

### Phase 7 — Teams and Scale

- Add organization membership, roles, and shared project spaces.
- Implement billing and usage enforcement if monetizing.
- Add Redis caching and scalable workers where justified.
- Introduce advanced observability and operational dashboards.
- Add enterprise controls and configurable retention policies.

## 18. Coding-Agent Execution Instructions

Use the following rules when delegating development to Claude Code, Codex, or another coding agent.

```markdown
# CogMem Development Instructions

You are implementing CogMem, a production-grade persistent
memory platform for AI agents.

## Rules

1. Inspect the existing repository before modifying anything.
2. Preserve working functionality and avoid unnecessary rewrites.
3. Read README.md, pyproject.toml, existing source files,
   tests, and configuration.
4. Follow the implementation roadmap in order.
5. Implement one independently testable milestone at a time.
6. Prefer small, modular changes over large rewrites.
7. Do not introduce unnecessary infrastructure or dependencies.
8. Do not require paid model APIs for core functionality.
9. Use typed Python, Pydantic v2, async-compatible APIs,
   and clear interfaces.
10. Write tests for every new feature and security boundary.
11. Run relevant tests, linting, and type checks after changes.
12. Never claim a test passed unless it was actually executed.
13. Never fabricate benchmark results or performance claims.
14. Update documentation and IMPLEMENTATION_STATUS.md
    after every completed milestone.

## Required completion report

After each milestone, report:

- Features implemented.
- Files added or modified.
- Architecture decisions and tradeoffs.
- Tests executed and their actual results.
- Known limitations and unresolved issues.
- Exact next milestone and prerequisites.

Do not implement future phases prematurely.
```

### Initial instruction to the coding agent

Start with a read-only repository assessment. Do not modify files yet.

```markdown
Inspect the existing CogMem repository.

Identify:
1. Current architecture and entry points.
2. Existing MCP tools and their implementation status.
3. Existing memory layers and storage backends.
4. Existing tests, dependencies, and deployment setup.
5. Security, correctness, and architectural risks.
6. Gaps against the Phase 1 requirements.

Produce:
- A concise architecture report.
- A file-by-file implementation plan.
- A list of changes required for Phase 1.
- A list of tests that will verify Phase 1.

Do not implement anything until the assessment is complete.
```

## 19. Definition of Done

CogMem's initial public MVP is ready when:

- [ ] A developer can install and run CogMem locally.
- [ ] An MCP-compatible agent can store and recall persistent memories.
- [ ] Memories survive server restarts and separate agent sessions.
- [ ] Retrieval returns relevant memories with provenance.
- [ ] Updates and deletions behave consistently.
- [ ] SQLite local mode and PostgreSQL hosted mode are supported.
- [ ] Tenant and memory-space isolation tests pass.
- [ ] Hosted MCP uses authenticated HTTPS connections.
- [ ] Documentation includes working client configuration examples.
- [ ] The project has automated tests, Docker deployment, and CI.
- [ ] Benchmark scripts can compare memory-enabled and memoryless agents without requiring a paid model API.

**Recommended build order:** Get the local MCP experience working end-to-end first, then make the memory engine reliable, then expose it as a hosted multi-tenant service. Add sophisticated graph reasoning, reflection, and scaling infrastructure only after the basic cross-session memory experience is demonstrably useful.
