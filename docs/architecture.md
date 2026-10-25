# Architecture

## Modules

| Module | Responsibility |
|---|---|
| `graphcite/extraction.py` | Prompt, JSON parsing and **grounding checks**: entity names must occur in the document; relation evidence must be a verbatim quote; endpoints must be accepted entities; self-loops and malformed items are dropped and reported. |
| `graphcite/models.py` | Dataclasses. Entity IDs are hashes of the normalised name, so the same entity merges across documents. Relation IDs hash source, type and target. |
| `graphcite/store.py` | `GraphStore` interface with `InMemoryGraphStore` and `Neo4jGraphStore`. Both return simple (cycle-free) undirected paths up to a validated depth and keep relation direction. |
| `graphcite/retrieval.py` | Entity matching (share of the entity's name words found in the question), then expansion and ranking: short paths first, then paths ending at another entity named in the question. Evidence is de-duplicated and numbered. |
| `graphcite/answer.py` | Answer prompt and **citation verification** (existence and word-containment support, threshold 0.5). |
| `graphcite/pipeline.py` | `ingest()` and `ask()`; refuses to call the model when nothing matches, so it cannot invent an answer from an empty graph. |
| `graphcite/llm.py` | `LLM` protocol and the Ollama implementation. |
| `graphcite/demo_data.py` | Fictional documents and a scripted model used by the no-services demo and by tests. |

## Data model (Neo4j)

- `(:Entity {id, name, type, summary, evidence})`, unique constraint on `id`.
- `(:Entity)-[:RELATED {id, type, evidence}]->(:Entity)`.
- `evidence` is a list of JSON strings, `{"doc": ..., "sentence": ...}`.
- Only `Entity` nodes are touched; other data in the database is left alone.

## Design decisions

- **Entities merge by normalised name across documents.** Per-document identities would make cross-document multi-hop reasoning impossible.
- **Depth is validated, then inlined into Cypher.** Cypher cannot parameterise variable-length path bounds, so depth must be an integer from 1 to 4.
- **One store contract, two implementations.** The same tests run against both, which keeps the in-memory demo honest about what Neo4j does.
- **The model never sees the whole graph**, only numbered evidence sentences, so every claim can be traced to a stored sentence.

## Known gaps

- Lexical matching and verification (no embeddings, no entailment).
- No coreference resolution: write or preprocess text with full entity names.
- Entity listing for matching is O(entities) per question.
