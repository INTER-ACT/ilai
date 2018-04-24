import unittest
import numpy as np
from gensim.models import KeyedVectors

import services.tagging.featureextraction as featureextraction
import services.tagging.preprocessing as preprocessing

class FeatureExtrTests(unittest.TestCase):

    def setUp(self):
        self.text = '2 mal am Tag besuche ich www.facebook.com'
        self.wörter = self.text.split(' ')
        featureextraction.create_wordvec(wordvec_path='../../../data/features/german_wv.model')
        self.wordvecs = featureextraction.getEmbeddings()

    def test_linkcount(self):
        self.assertEqual(featureextraction.link_vec(self.text), 1)
    def test_numberCount(self):
        self.assertEqual(len(featureextraction.number_count(self.text)), 1)
    def test_getVecsforText(self):
        finaltext = preprocessing.remove_non_vocab(words=self.wörter, wordvecs=self.wordvecs)
        self.assertTrue(np.array_equal(np.shape(featureextraction.get_vecsForTexts([finaltext])[0]), np.shape(np.empty([4,300]))))
    def test_getWordVecforWord(self):
        print(np.shape(featureextraction.get_Wordvec(self.wörter[0])))
        self.assertTrue(np.array_equal(np.shape(featureextraction.get_Wordvec(self.wörter[0])),np.shape(np.empty([300,]))))

if __name__ == '__main__':
    unittest.main()
