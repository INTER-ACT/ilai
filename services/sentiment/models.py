from os import path
import numpy as np

import services.sentiment.classification as classification
import services.sentiment.pipelines as pipelines
from exceptions.model_exceptions import LoadModelException, SaveModelException
from services.abstract_models import AModel
from services.sentiment.feature_extraction import get_wv_keras_embedding
from services.train_utility import encode_one_hot, get_solution_vec


class LSTMPaddedModel(AModel):
    def __init__(self, model_name, tags, save_dir='./models_saves', log_dir='./tb', tf_graph=None, sequence_length=25,
                 hidden_layer_sizes=[100], dropouts=[0.3], activations=['relu']):
        self.sequence_length = sequence_length
        clf = classification.LSTMPadded(log_dir=log_dir + '/' + model_name, sequence_length=self.sequence_length,
                                        embedding_layer=get_wv_keras_embedding(), tf_graph=tf_graph,
                                        hidden_layer_sizes=hidden_layer_sizes,
                                        dropouts=dropouts, activations=activations)
        super().__init__(clf, model_name, tags, save_dir)

    def predict(self, texts, threshold=0, return_predict_vector=False):
        X = pipelines.get_embedding_indices(texts, maxlen=self.sequence_length)
        predict_vec = self.clf.predict(X)
        if return_predict_vector:
            return predict_vec

        predictions = []
        for prediction in predict_vec.argmax(axis=1):
            predictions.append([self.tags[prediction]])
        return predictions

    def train(self, texts, solutions):
        Y = encode_one_hot(get_solution_vec(solutions, tags=self.tags))
        X = pipelines.get_embedding_indices(texts, maxlen=self.sequence_length)
        self.clf.train(X, Y)

    def test(self, texts, solutions, avg_metrics=False):
        Y = encode_one_hot(get_solution_vec(solutions, self.tags))
        X = pipelines.get_embedding_indices(texts, maxlen=self.sequence_length)
        return super()._evaluate_metrics(X, Y, avg_metrics)


class LSTMSequenceModel(AModel):
    def __init__(self, model_name, tags, save_dir='./models_saves', log_dir='./tb', tf_graph=None,
                 hidden_layer_sizes=[100], dropouts=[0.0], activations=['relu'], ):
        self.sequence_length = 25
        clf = classification.LSTMSequence(log_dir=log_dir + '/' + model_name, tf_graph=tf_graph,
                                          hidden_layer_sizes=hidden_layer_sizes, dropouts=dropouts,
                                          activations=activations)
        super().__init__(clf, model_name, tags, save_dir)

    def predict(self, texts, threshold=0, return_predict_vector=False):
        X = pipelines.get_wv_vec_sequence(texts)
        predict_vec = self.clf.predict(X)
        if return_predict_vector:
            return predict_vec

        predictions = []
        for prediction in predict_vec.argmax(axis=1):
            predictions.append(self.tags[prediction])
        return predictions

    def train(self, texts, solutions):
        Y = encode_one_hot(get_solution_vec(solutions, tags=self.tags))
        X = pipelines.get_wv_vec_sequence(texts)
        self.clf.train(X, Y)

    def test(self, texts, solutions, avg_metrics=False):
        Y = encode_one_hot(get_solution_vec(solutions, self.tags))
        X = pipelines.get_wv_vec_sequence(texts)
        return super()._evaluate_metrics(X, Y, avg_metrics)


class CNNModel(AModel):
    def __init__(self, model_name, tags, save_dir='./models_saves', log_dir='./tb', tf_graph=None, sequence_length=40,
                 filter_sizes=[[3, 4, 5]], num_filters=[128], paddings=['same'], conv_activations=['relu'],
                 dense_layer_sizes=[128, 3], dense_activations=['relu', 'softmax'], dropout_rate=0.5):
        self.sequence_length = sequence_length
        clf = classification.CNN(log_dir=log_dir + '/' + model_name, sequence_length=self.sequence_length,
                                 embedding_layer=get_wv_keras_embedding(), tf_graph=tf_graph, filter_sizes=filter_sizes,
                                 num_filters=num_filters, paddings=paddings, conv_activations=conv_activations,
                                 dense_layer_sizes=dense_layer_sizes, dense_activations=dense_activations,
                                 dropout_rate=dropout_rate)
        super().__init__(clf, model_name, tags, save_dir)

    def predict(self, texts, threshold=0, return_predict_vector=False):
        X = pipelines.get_embedding_indices(texts, maxlen=self.sequence_length)
        predict_vec = self.clf.predict(X)
        if return_predict_vector:
            return predict_vec

        predictions = []
        for prediction in predict_vec.argmax(axis=1):
            predictions.append(self.tags[prediction])
        return predictions

    def train(self, texts, solutions):
        try:
            Y = encode_one_hot(get_solution_vec(solutions, self.tags))
            X = pipelines.get_embedding_indices(texts, maxlen=self.sequence_length)
            self.clf.train(X, Y)
        except RuntimeError:
            print("Training model failed")
            raise

    def test(self, texts, solutions, avg_metrics=False):
        Y = encode_one_hot(get_solution_vec(solutions, self.tags))
        X = pipelines.get_embedding_indices(texts, maxlen=self.sequence_length)
        return super()._evaluate_metrics(X, Y, avg_metrics)


class DNNModel(AModel):
    def __init__(self, model_name, tags, save_dir='./models_saves', log_dir='./tb', tf_graph=None,
                 hidden_layer_sizes=[100, 100], activations=['relu', 'relu', 'softmax']):
        clf = classification.DNN(log_dir=log_dir + '/' + model_name, tf_graph=tf_graph,
                                 hidden_layer_sizes=hidden_layer_sizes, activations=activations)
        super().__init__(clf, model_name, tags, save_dir)

    def predict(self, texts, threshold=0, return_predict_vector=False):
        X = pipelines.get_feature_vec(texts)
        predict_vec = self.clf.predict(X)
        if return_predict_vector:
            return predict_vec

        predictions = []
        for prediction in predict_vec.argmax(axis=1):
            predictions.append(self.tags[prediction])
        return predictions

    def train(self, texts, solutions):
        Y = encode_one_hot(get_solution_vec(solutions, self.tags))
        X = pipelines.get_feature_vec(texts)
        self.clf.train(X, Y)

    def test(self, texts, solutions, avg_metrics=False):
        Y = encode_one_hot(get_solution_vec(solutions, self.tags))
        X = pipelines.get_feature_vec(texts)
        return super()._evaluate_metrics(X, Y, avg_metrics)


class SVMModel(AModel):
    def __init__(self, model_name, tags, save_dir='./models_saves', log_dir='./tb'):
        clf = classification.SVMKernelClassifier(log_dir=log_dir + '/' + model_name)
        super().__init__(clf, model_name, tags, save_dir)

    def predict(self, texts, threshold=0, return_predict_vector=False):
        X = pipelines.get_feature_vec(texts)
        predict_vec = self.clf.predict(X)
        if return_predict_vector:
            return predict_vec

        predictions = []
        for prediction in predict_vec:  # sklearn returns list of predictions
            predictions.append(self.tags[prediction])
        return predictions

    def train(self, texts, solutions):
        Y = get_solution_vec(solutions, self.tags)
        X = pipelines.get_feature_vec(texts)
        self.clf.train(X, Y)

    def test(self, texts, solutions, avg_metrics=False):
        Y = get_solution_vec(solutions, self.tags)
        X = pipelines.get_feature_vec(texts)
        return super()._evaluate_metrics(X, Y, avg_metrics)


class CombinedModel(AModel):
    # tags = ['NEUTRAL', 'CONTRA', 'PRO']
    def __init__(self, model_name, tags, save_dir='./models_saves', log_dir='./tb', tf_graph=None):
        self.sequence_length = 25
        self.clf_argumentative = classification.LSTMPadded(log_dir=log_dir + '/' + model_name,
                                                           sequence_length=self.sequence_length,
                                                           embedding_layer=get_wv_keras_embedding(), tf_graph=tf_graph)
        self.clf_stance = classification.LSTMPadded(log_dir=log_dir + '/' + model_name,
                                                    sequence_length=self.sequence_length,
                                                    embedding_layer=get_wv_keras_embedding(), tf_graph=tf_graph)
        super().__init__(None, model_name, tags, save_dir)

    def predict(self, texts, threshold=0, return_predict_vector=False):
        X = pipelines.get_wv_vec(texts, self.sequence_length)
        arg_prediction_vec = self.clf_argumentative.predict(X)
        stance_prediction_vec = self.clf_stance.predict(X)
        if return_predict_vector:
            return arg_prediction_vec, stance_prediction_vec

        predictions = []
        for arg_pred, stance_pred in zip(arg_prediction_vec.argmax(axis=1), stance_prediction_vec.argmax(axis=1)):
            if arg_pred == 1:
                predictions.append(self.tags[0])
            else:
                predictions.append(self.tags[stance_pred + 1])
        return predictions

    def train(self, texts, solutions):
        X = pipelines.get_wv_vec(texts, self.sequence_length)
        solution_vec = get_solution_vec(solutions, tags=self.tags)
        stance_indices = np.where(solution_vec > 0)  # NEUTRAL = 0

        X_stance = X[stance_indices]
        stance_solution = solution_vec[stance_indices]
        stance_solution -= 1
        Y_stance = encode_one_hot(stance_solution)

        np.place(solution_vec, solution_vec == 2, 1)
        Y_arg = encode_one_hot(solution_vec)

        self.clf_argumentative.train(X, Y_arg)
        self.clf_stance.train(X_stance, Y_stance)

    def test(self, texts, solutions, avg_metrics=False):
        X = pipelines.get_wv_vec(texts, self.sequence_length)
        solution_vec = get_solution_vec(solutions, tags=self.tags)
        stance_indices = np.where(solution_vec > 0)  # NEUTRAL = 0

        X_stance = X[stance_indices]
        stance_solution = solution_vec[stance_indices]
        stance_solution -= 1
        Y_stance = encode_one_hot(stance_solution)

        np.place(solution_vec, solution_vec == 2, 1)
        Y_arg = encode_one_hot(solution_vec)

        self.clf_argumentative.train(X, Y_arg)
        self.clf_stance.train(X_stance, Y_stance)

        X = pipelines.get_wv_vec(texts, self.sequence_length)
        arg_prediction_vec = self.clf_argumentative.predict(X)
        stance_prediction_vec = self.clf_stance.predict(X)

        return None

    def load(self, file_appendix):
        load_file_stance = self.name + '_stance' + '_' + file_appendix
        load_path_stance = path.join(self.save_dir, load_file_stance)
        load_file_arg = self.name + '_stance' + '_' + file_appendix
        load_path_arg = path.join(self.save_dir, load_file_arg)
        try:
            self.clf_stance.load(load_path_stance)
            self.clf_argumentative.load(load_path_arg)
        except LoadModelException:
            raise

    def save(self, file_appendix):
        save_file_stance = self.name + '_stance' + '_' + file_appendix
        save_path_stance = path.join(self.save_dir, save_file_stance)
        save_file_arg = self.name + '_stance' + '_' + file_appendix
        save_path_arg = path.join(self.save_dir, save_file_arg)
        try:
            self.clf_stance.save(save_path_stance)
            self.clf_argumentative.save(save_path_arg)
        except SaveModelException:
            raise
