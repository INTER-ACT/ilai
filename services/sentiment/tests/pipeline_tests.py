import unittest

import numpy as np

import services.sentiment.pipelines as pipelines


class PipelineTests(unittest.TestCase):

    def setUp(self):
        self.sentence = 'Ich bedanke  mich für eure 27 #Beiträge und 8 Ideen, habe was dazugelernt! '
        self.text = 'nur spitze, und super witzig kann ich DIESES Argument bewerten!'
        pipelines.init_pipelines()

    def test_pipe_feates_before_preprocessing(self):
        print(pipelines.pipe_features_before_preprocessing(self.text))
        self.assertTrue(np.array_equal(pipelines.pipe_features_before_preprocessing(self.text), [[0, 1, 0, 0, 1, 1, 0, 1, 0, 0]]))

    def test_pipe_preprocessing(self):
        self.assertEqual(pipelines.pipe_preprocessing(self.text), 'spitze super witzig Argument bewerten')
        self.assertEqual(pipelines.pipe_preprocessing(self.sentence), 'bedanke fuer Beitraege Ideen dazugelernt')

if __name__ == '__main__':
    unittest.main()