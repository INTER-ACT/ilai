import unittest
from gensim.models import KeyedVectors
import numpy as np

import services.tagging.preprocessing as preprocessing
from config import Config



class Prepr(unittest.TestCase):
    config = Config()
    taglist = ['User-Generated-Content', 'wirtschaftliche Interessen', 'Download & Streaming',
                    'Rechteinhaberschaft',
                    'Respekt & Anerkennung', 'Freiheiten der Nutzer', 'Bildung & Wissenschaft', 'kulturelles Erbe',
                    'soziale Medien', 'Nutzung fremder Inhalte']
    stopwords_path = '../../../data/features/stopwords-de.json'
    csv_path = '../../../data/training_data/Tags_Kommentare.csv'
    testsatz ='Dieser  Testsatz soll ä 3123 die PrE?processing Unit testen'
    vector_path = '../../../data/features/german_wv.model'
    preprocessing.init_stopwords(stopwords_path=stopwords_path)
    preprocessing.init_taglist(taglist)
    def test_removeStops(self):
        self.assertEqual(preprocessing.remove_stopwords(self.testsatz), "Dieser Testsatz ä 3123 PrE?processing Unit testen")
    def test_removeSpaces(self):
        self.assertEqual(preprocessing.remove_spaces(self.testsatz), "Dieser Testsatz soll ä 3123 die PrE?processing Unit testen")

    def test_replaceUmlaute(self):
        self.assertEqual(preprocessing.replace_umlaute(self.testsatz), "Dieser  Testsatz soll ae 3123 die PrE?processing Unit testen")
    def test_removeSatzSonder(self):
        self.assertEqual(preprocessing.remove_satzsonder(self.testsatz),"Dieser Testsatz soll 3123 die PrE processing Unit testen")
    def test_lemmatization(self):
        text = ' '.join(preprocessing.lemmatization(self.testsatz))
        self.assertEqual(text,"dies Testsatz sollen ä 3123 die Pre processing Unit testen")
    def test_toLower(self):
        self.assertEqual(preprocessing.tolower(self.testsatz), "dieser  testsatz soll ä 3123 die pre?processing unit testen")
    def test_removeNonVocab(self):
        wordvecs = KeyedVectors.load_word2vec_format(self.vector_path,binary=True)
        text = self.testsatz
        wörter = text.split(' ')
        self.assertEqual(preprocessing.remove_non_vocab(wörter, wordvecs), "Dieser 3123 Unit testen")


    def test_getOneHotEncoding(self):
        text = self.testsatz
        solution = 'soziale Medien'
        texts = []
        texts.append(texts)
        solutions = []
        solutions.append(solution)
        oneHot=preprocessing.oneHotEncoding(texts,solutions)[0]
        print(oneHot)
        y=[]
        y.append(np.zeros(10, dtype=float))
        y[0][8] = 1
        print(y[0])
        self.assertTrue(np.array_equal(oneHot,y[0]))

    def test_preprocessingPipe(self):
        wordvecs = KeyedVectors.load_word2vec_format(self.vector_path,binary=True)  # falls Datei schon exisitert / binary = C binary format
        text = self.testsatz
        self.assertEqual(preprocessing.preprocessing_pipeline(text,wordvecs),'ae 3123 Pre processing Unit testen')

if __name__ == '__main__':
    unittest.main()