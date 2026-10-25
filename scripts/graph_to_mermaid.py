"""Print the bundled demo graph as a Mermaid flowchart:  python scripts/graph_to_mermaid.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphcite import pipeline  # noqa: E402
from graphcite.demo_data import DOCUMENTS, ScriptedLLM  # noqa: E402
from graphcite.store import InMemoryGraphStore  # noqa: E402

store, llm = InMemoryGraphStore(), ScriptedLLM()
for name, text in DOCUMENTS.items():
    pipeline.ingest(store, llm, name, text)
entities = {e.id: e for e in store.all_entities()}
print("flowchart LR")
for e in entities.values():
    print(f'    {e.id}["{e.name}<br/><i>{e.type}</i>"]')
for r in store.all_relations():
    print(f"    {r.source_id} -->|{r.type}| {r.target_id}")
