import unittest
import pandas as pd
import numpy as np

import services.tagging.models as Catlyn
from exceptions.model_exceptions import SaveModelException, LoadModelException
import services.tagging.preprocessing as  preprocessing
import services.tagging.featureextraction as featureextraction


class DataFileTest(unittest.TestCase):
    def setUp(self):
        self.taglist = ['User-Generated-Content', 'wirtschaftliche Interessen', 'Download & Streaming', 'Rechteinhaberschaft',
           'Respekt & Anerkennung', 'Freiheiten der Nutzer', 'Bildung & Wissenschaft', 'kulturelles Erbe',
           'soziale Medien', 'Nutzung fremder Inhalte']
        self.csv_path='../../../data/training_data/'
        self.modelspath = '../../../data/model_saves'
        self.wordvecs = '../../../data/features/german_wv.model'
        self.stopwords_path = '../../../data/features/stopwords-de.json'
        self.model = Catlyn.LSTMOneHot('lstm_model', tags=self.taglist, save_dir=self.modelspath)

        preprocessing.init_taglist(self.taglist)
        preprocessing.init_stopwords(stopwords_path=self.stopwords_path)
        featureextraction.create_wordvec(wordvec_path=self.wordvecs)

    def test_loadCorruptModel(self):
        self.assertRaises(LoadModelException, lambda: self.model.load('corrupted'))

    def test_DeleteATag(self):
        df = pd.read_csv(self.csv_path+'Tags_Kommentare.csv', skipinitialspace=True, sep=';', encoding='utf-8')
        X = []
        Y = []
        for index, row in df.iterrows():
            tag = str(row['Tag'])
            if tag in self.taglist:
                X.append(str(row['Kommentar']))
                Y.append(tag)
        self.assertRaises(RuntimeError, lambda: self.model.train(X,Y))

    def test_getData(self):
        df = pd.read_csv(self.csv_path+'datafileTests.csv', skipinitialspace=True, sep=';', encoding='utf-8')
        X = []
        Y = []
        for index, row in df.iterrows():
            tag = str(row['Tag'])
            if tag in self.taglist:
                X.append(str(row['Kommentar']))
                Y.append(tag)
        self.assertEqual(len(X),3)


if __name__ == '__main__':
    unittest.main()