import unittest
import pandas as pd
import numpy as np

import services.tagging.models as Catlyn
from exceptions.model_exceptions import SaveModelException, LoadModelException
import services.sentiment.preprocessing as  preprocessing
import services.sentiment.feature_extraction as featureextraction
import services.sentiment.pipelines as pipelines
from services.sentiment.models import LSTMPaddedModel


class DataFileTest(unittest.TestCase):
    def setUp(self):
        self.corpus_path = '../../../data/training_data/trainingsdaten_sentiment_stance_equal_size.csv'
        pipelines.init_pipelines()
        self.positive_tags = ['ARGUMENTATIVE', 'PRO']
        self.negative_tags = ['NON-ARGUMENTATIVE', 'CONTRA', 'NEUTRAL']

        self.modelspath = '../../../data/model_saves'

        self.model = LSTMPaddedModel(model_name='LSTM_padded', tags=["PRO"],
                        save_dir=self.modelspath)

    def test_loadCorruptModel(self):
        self.assertRaises(LoadModelException,lambda: self.model.load('corrupted'))
    def test_DeleteATag(self):
        df = pd.read_csv(self.corpus_path, skipinitialspace=True, sep=';', encoding='utf-8', header=None,
                         names=['Tag', 'Text'])
        df = df.iloc[np.random.permutation(len(df))]  # shuffle training set
        df = df.reset_index(drop=True)
        X = df['Text'].tolist()
        Y = df['Tag'].tolist()
        self.assertRaises(RuntimeError,lambda: self.model.train(X,Y))

    def test_getData(self):
        self.corpus_path = '../../../data/training_data/datafileTests.csv'
        df = pd.read_csv(self.corpus_path, skipinitialspace=True, sep=';', encoding='utf-8', header=None,
                         names=['Tag', 'Text'])
        df = df.iloc[np.random.permutation(len(df))]  # shuffle training set
        df = df.reset_index(drop=True)
        self.assertEquals(len(df['Tag'].tolist()),4)


if __name__ == '__main__':
    unittest.main()