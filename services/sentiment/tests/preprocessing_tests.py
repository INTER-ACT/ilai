import unittest

import services.sentiment.preprocessing as preprocessing


class PreprocessingTests(unittest.TestCase):

    def setUp(self):
        self.sentence = 'Ich bedanke  mich für eure 27 #Beiträge und 8 Ideen, habe was dazugelernt! '
        preprocessing.init_stop_words('../data/features/stopwords-de.json')

    def test_remove_numbers(self):
        self.assertEqual(preprocessing.remove_numbers(self.sentence),
                         'Ich bedanke  mich für eure  #Beiträge und  Ideen, habe was dazugelernt! ')

    def test_remove_punctuation(self):
        self.assertEqual(preprocessing.remove_punctuation(self.sentence),
                         'Ich bedanke  mich für eure 27 #Beiträge und 8 Ideen habe was dazugelernt ')

    def test_remove_special_chars(self):
        self.assertEqual(preprocessing.remove_punctuation(self.sentence),
                         'Ich bedanke  mich für eure 27 Beiträge und 8 Ideen habe was dazugelernt ')

    def test_minimizing_space(self):
        self.assertEqual(preprocessing.minimizing_space(self.sentence),
                         'Ich bedanke mich für eure 27 #Beiträge und 8 Ideen, habe was dazugelernt!')

    def test_remove_stop_words(self):
        self.assertEqual(preprocessing.remove_stop_words(self.sentence),
                         'bedanke 27 Beiträge 8 Ideen dazugelernt')

    def test_replace_umlauts(self):
        self.assertEqual(preprocessing.replace_umlauts(self.sentence),
                         'Ich bedanke  mich fuer eure 27 #Beitraege und 8 Ideen, habe was dazugelernt! ')

    def test_lemmatize(self):
        self.assertEqual(preprocessing.lemmatize(self.sentence),
                         'ich bedanke mich für eure 27 Beiträge und 8 Ideen haben was dazugelernt')


if __name__ == '__main__':
    unittest.main()
