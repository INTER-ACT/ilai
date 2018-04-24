import unittest
import numpy as np
from gensim.models import KeyedVectors

import services.tagging.models as Catlyn
import services.tagging.featureextraction as featureextraction
import services.tagging.preprocessing as preprocessing



class FeatureExtrTests(unittest.TestCase):

    def setUp(self):
        self.text = 'Das Downloaden von Filmen ist illegal und kann Folgen mit sich ziehen'
        self.taglist = ['User-Generated-Content', 'wirtschaftliche Interessen', 'Download & Streaming', 'Rechteinhaberschaft',
           'Respekt & Anerkennung', 'Freiheiten der Nutzer', 'Bildung & Wissenschaft', 'kulturelles Erbe',
           'soziale Medien', 'Nutzung fremder Inhalte']
        preprocessing.init_taglist(tags=self.taglist)
        self.model = Catlyn.LSTMOneHot('lstm_model',tags=self.taglist,save_dir='../../../data/model_saves')
        stopwords_path = '../../../data/features/stopwords-de.json'
        preprocessing.init_stopwords(stopwords_path=stopwords_path)


        featureextraction.create_wordvec(wordvec_path='../../../data/features/german_wv.model')

    def test_classification(self):
        self.model.load('model7')
        print(self.model.predict([self.text],60))
        self.assertEqual(self.model.predict([self.text],60)[0][0],'Download und Streaming')
        #finaltext = preprocessing.remove_non_vocab(words=self.wörter, wordvecs=self.wordvecs)
        #self.assertTrue(np.array_equal(np.shape(featureextraction.get_vecsForTexts([finaltext])[0]), np.shape(np.empty([4,300]))))

if __name__ == '__main__':
    unittest.main()
