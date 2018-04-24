import numpy as np

import services.tagging.featureextraction as featureextraction
import services.tagging.preprocessing as preprocessing
import services.tagging.classification as classification
from services.abstract_models import AModel
from exceptions.model_exceptions import LoadModelException, SaveModelException, RunTimeException


class LSTMOneHot(AModel):
    def __init__(self, model_name, tags, save_dir='services/tagging/model_saves/', log_dir='./tb', tf_graph=None,layer_sizes=[100,100],activation=['tanh','tanh'],optimizer='adam',loss='mean_squared_error',dropout_rate=[0.2,0.2]):
        classifier = classification.LSTMOneHotLabels(model_name=model_name, tf_graph=tf_graph,layer_sizes=layer_sizes,activation=activation,optimizer=optimizer,loss=loss,dropout_rate=dropout_rate)
        super().__init__(classifier, model_name, tags=tags, save_dir=save_dir)
        self.init_weights = self.clf.model.get_weights()


    def predict(self, texts, threshold=50, predict_vector=False):
        X = featureextraction.get_vecsForTexts(preprocessing.preprocess_texts(texts, featureextraction.getEmbeddings()))
        predictions = self.clf.predict(X)
        if predict_vector == True:
            return predictions
        else:
            tagpredictions = []
            for _, prediction in enumerate(predictions):
                taglist = []
                for idx, element in enumerate(prediction):
                    if threshold / 100 <= element:
                        taglist.append(self.tags[idx])
                tagpredictions.append(taglist)
            return tagpredictions

    def train(self, texts, solutions):
        if texts is not None and solutions is not None:
            X = featureextraction.get_vecsForTexts(
                preprocessing.preprocess_texts(texts, featureextraction.getEmbeddings()))
            Y = preprocessing.oneHotEncoding(texts, solutions)
            try:
                self.clf.train(X, Y)
            except RunTimeException as RunTimeExc:
                print("[MODEL] {}".format(RunTimeExc.args[0]))
                raise

    def reset(self):
        self.clf.reset()

    #def test(self, texts, solutions):
    #    if texts is not None and solutions is not None:
    #        X = featureextraction.get_vecsForTexts(
    #            preprocessing.preprocess_texts(texts, featureextraction.getEmbeddings()))
    #        Y = preprocessing.oneHotEncoding(texts, solutions)
    #        metrics = []
    #        for index in range(len(self.tags)):
    #            x = np.array(X)
    #            y = np.array(Y)
    #            metrics.append(super()._evaluate_metrics(x, y, category='{}'.format(index)))
    #        return metrics

    def test(self, texts, solutions, avg_metrics=False):
        Y = preprocessing.oneHotEncoding(texts,solutions)
        X = featureextraction.get_vecsForTexts(preprocessing.preprocess_texts(texts,featureextraction.getEmbeddings()))
        return super()._evaluate_metrics(X, Y, avg_metrics)
