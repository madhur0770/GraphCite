import unittest

from graphcite.text import containment, find_sentence_containing, normalize_name, split_sentences, tokenize


class TextTests(unittest.TestCase):
    def test_normalize_name_ignores_case_punctuation_and_possessive(self):
        self.assertEqual(normalize_name("Aurora  Labs"), normalize_name("aurora labs"))
        self.assertEqual(normalize_name("Maria Santos's"), "maria santos")

    def test_tokenize_keeps_hyphenated_words(self):
        self.assertEqual(tokenize("State-of-the-art, 2018."), ["state-of-the-art", "2018"])

    def test_split_sentences(self):
        self.assertEqual(split_sentences("One is here. Two follows! Is three? Yes."), ["One is here.", "Two follows!", "Is three?", "Yes."])

    def test_containment_bounds(self):
        self.assertEqual(containment("Aurora Labs was founded by Maria Santos", "Aurora Labs was founded by Maria Santos in 2018."), 1.0)
        self.assertEqual(containment("Quantum pancakes", "Aurora Labs is in Lisbon."), 0.0)
        self.assertEqual(containment("the of and", "anything"), 0.0)

    def test_find_sentence_is_case_and_whitespace_insensitive(self):
        doc = "First sentence here. Second   sentence\nis this one. Third."
        self.assertEqual(find_sentence_containing(doc, "SECOND sentence is this"), "Second sentence is this one.")
        self.assertIsNone(find_sentence_containing(doc, "not present"))
        self.assertIsNone(find_sentence_containing(doc, "   "))


if __name__ == "__main__":
    unittest.main()
