import unittest
import numpy as np
from textblob_de import TextBlobDE as TextBlob

from config import Config
import services.sentiment.feature_extraction as feature_extraction
from services.sentiment.preprocessing import remove_non_vocab_words


class FeatureExtractionTests(unittest.TestCase):

    def setUp(self):
        config = Config()
        self.text = 'nur spitz, und super witzig kann ich DIESES Argument bewerten!'
        feature_extraction.init_subjectivity_clues(config.paths['subjectivity_clues'])
        feature_extraction.init_sepl(config.paths['sepl'])
        feature_extraction.init_polarity_clues(config.paths['polarity_clues'])
        feature_extraction.init_word_vectors(config.paths['word_vector_model'])

    def test_caps_words_count(self):
        self.assertEqual(feature_extraction.get_caps_words_count(self.text), [[1]])

    def test_punctuation_vector(self):
        self.assertTrue(np.array_equal(feature_extraction.get_punctuation_vector(self.text), [[0, 1, 0, 0, 1]]))

    def test_last_char_vector(self):
        self.assertTrue(np.array_equal(feature_extraction.get_last_char_vector(self.text), [[0, 1, 0, 0]]))

    def test_sentiment_phrase_score(self):
        self.assertEqual(feature_extraction.get_sentiment_phrase_score(self.text), [[0.924 + 0.4 + 0.893]])

    def test_polarity_score(self):
        self.assertEqual(feature_extraction.get_polarity_score(self.text), [[3]])

    def test_subjectivity_score(self):
        self.assertEqual(feature_extraction.get_subjectivity_score(self.text), [[0]])

    def test_word_vectors(self):
        text = remove_non_vocab_words(self.text, feature_extraction.word_vectors.vocab)
        blob = TextBlob(text)
        word_vecs = feature_extraction.get_word_vec_repr(blob.words)
        self.assertEqual(len(word_vecs), 5)
        self.assertEqual(len(word_vecs[0]), 300)


if __name__ == '__main__':
    unittest.main()
