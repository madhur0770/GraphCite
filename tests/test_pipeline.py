import unittest

from graphcite import pipeline
from graphcite.demo_data import DOCUMENTS, SAMPLE_QUESTIONS, ScriptedLLM
from graphcite.extraction import parse_extraction
from graphcite.store import InMemoryGraphStore


class FakeLLM:
    """Records prompts; returns fixed text."""

    def __init__(self, answer):
        self.answer, self.prompts = answer, []

    def complete_json(self, prompt):  # pragma: no cover - unused here
        raise AssertionError("not expected")

    def complete(self, prompt):
        self.prompts.append(prompt)
        return self.answer


def demo_store():
    store, llm = InMemoryGraphStore(), ScriptedLLM()
    for name, text in DOCUMENTS.items():
        pipeline.ingest(store, llm, name, text)
    return store, llm


class DemoDataTests(unittest.TestCase):
    def test_every_scripted_extraction_is_fully_grounded(self):
        llm = ScriptedLLM()
        for name, text in DOCUMENTS.items():
            result = parse_extraction(llm.complete_json(text), name, text)
            self.assertEqual(result.dropped, [], f"{name}: {result.dropped}")
            self.assertGreater(len(result.relations), 0)


class PipelineTests(unittest.TestCase):
    def test_entities_merge_across_documents(self):
        store, _ = demo_store()
        aurora = next(e for e in store.all_entities() if e.name == "Aurora Labs")
        self.assertEqual({ev.doc for ev in aurora.evidence}, {"aurora-labs-profile", "maria-santos-bio"})
        self.assertEqual(store.stats(), {"entities": 8, "relations": 7, "documents": 3})

    def test_two_hop_question_cites_two_documents(self):
        store, llm = demo_store()
        result = pipeline.ask(store, llm, *SAMPLE_QUESTIONS[0])
        self.assertIn("Maria Santos", result.text)
        self.assertEqual(result.supported_ratio, 1.0)
        self.assertIn("aurora-labs-profile", {i.doc for i in result.evidence})

    def test_three_hop_question_reaches_the_third_document(self):
        store, llm = demo_store()
        question, depth = SAMPLE_QUESTIONS[1]
        shallow = pipeline.ask(store, llm, question, 1)
        deep = pipeline.ask(store, llm, question, depth)
        self.assertNotIn("Ionex", " ".join(i.sentence for i in shallow.evidence))
        self.assertIn("Volta Systems acquired Ionex in 2021.", [i.sentence for i in deep.evidence])
        self.assertEqual(deep.supported_ratio, 1.0)

    def test_unanswerable_question_does_not_call_the_model(self):
        store, _ = demo_store()
        llm = FakeLLM("should not be used")
        result = pipeline.ask(store, llm, *SAMPLE_QUESTIONS[2])
        self.assertEqual(llm.prompts, [])
        self.assertEqual(result.evidence, [])
        self.assertIn("nothing to answer", result.text)

    def test_model_that_ignores_citation_rules_is_flagged(self):
        store, _ = demo_store()
        llm = FakeLLM("Maria Santos founded Aurora Labs. Helios Drive is cheap [9].")
        result = pipeline.ask(store, llm, "Who founded Aurora Labs and what is Helios Drive?", 2)
        self.assertEqual([c.status for c in result.claims], ["uncited", "invalid-citation"])
        self.assertEqual(result.supported_ratio, 0.0)
        self.assertEqual(result.cited_refs, [])

    def test_prompt_contains_numbered_evidence_and_citation_instruction(self):
        store, _ = demo_store()
        llm = FakeLLM("Aurora Labs was founded by Maria Santos in 2018 [1].")
        pipeline.ask(store, llm, "Who founded Aurora Labs?", 1)
        prompt = llm.prompts[0]
        self.assertIn("[1] ", prompt)
        self.assertIn("End EVERY sentence with the number(s) of the evidence", prompt)
        self.assertIn("ONLY the numbered evidence", prompt)

    def test_ingest_of_unknown_text_with_scripted_model_adds_nothing(self):
        store, llm = InMemoryGraphStore(), ScriptedLLM()
        result = pipeline.ingest(store, llm, "x", "Some entirely new text about Zorblax Industries.")
        self.assertEqual((result.entities, result.relations), ([], []))
        self.assertEqual(store.stats()["entities"], 0)


if __name__ == "__main__":
    unittest.main()
