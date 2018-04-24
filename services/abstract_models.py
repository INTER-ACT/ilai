import os.path
from abc import ABCMeta, abstractmethod

from exceptions.model_exceptions import LoadModelException, SaveModelException
from services.train_utility import calculate_metrics


class AClassifier(metaclass=ABCMeta):

    @abstractmethod
    def predict(self, X):
        """
        :param X: np.array([m, n])  n...features
        :return: np.array([m, N]    N...classes
        """
        pass

    @abstractmethod
    def train(self, X, Y):
        """
        :param X: np.array([m, n])
        :param Y: np.array([m, N]
        :return: None
        """
        pass

    @abstractmethod
    def save(self, path):
        """
        :param path: path to save model config to
        :return: bool successful
        """
        pass

    @abstractmethod
    def load(self, path):
        """
        :param path: path to load model config from
        :return: bool successful
        """
        pass

    def check_load(self, path):
        path += self.file_extension
        if not os.path.isfile(path):
            raise LoadModelException("Loading model failed. File {} not found.".format(path))


class AModel(metaclass=ABCMeta):
    def __init__(self, clf, model_name, tags, save_dir='./data/models_saves'):
        self.clf = clf
        self.name = model_name
        self.tags = tags
        self.save_dir = save_dir

    @abstractmethod
    def predict(self, texts, threshold, return_predict_vector=False):
        """
        Return predicted tags of texts
        :param texts: [string]
        :param threshold: min certainty of prediction
        :param return_predict_vector: if true the model returns the vector from the classifier, if false it returns the predicted tags as strings
        :return: tags [string] or predict vector: np.array
        """
        pass

    @abstractmethod
    def train(self, texts, solutions):
        """
        Train model with texts
        :param texts: [string]
        :param solutions: tags [string]
        :return: None
        """
        pass

    @abstractmethod
    def test(self, texts, solutions, avg_metrics=False):
        """
        Return metrics of model on texts
        :param texts: [string]
        :param solutions: [string]
        :param avg_metrics: bool - when calling test over api metrics are averaged
        :return: metrics dict
        """
        pass

    def _evaluate_metrics(self, X, Y, avg_metrics=False, print_metrics=False):
        """
        Return and/or print metrics of model on test set
        :param X: np.array([m, n]) - features
        :param Y: np.array([m, N]) - solutions
        :param print_metrics: boolean
        :return: metrics dict
        """
        metrics = calculate_metrics(self.clf, X, Y, avg_metrics=avg_metrics)
        if print_metrics:
            print(
                'ID: {}\nAccuracy: {} -- {}\nf1: {} -- {}\nprecision: {} -- {}\nrecall: {} -- {}\nconfusion-matrix:\n{}'
                    .format(self.clf.id, metrics['accuracy'], metrics['accuracy'].sum() / len(metrics['accuracy']),
                            metrics['f1'], metrics['f1'].sum() / len(metrics['f1']),
                            metrics['precision'], metrics['precision'].sum() / len(metrics['precision']),
                            metrics['recall'], metrics['recall'].sum() / len(metrics['recall']),
                            metrics['confusion_matrix']))
            return metrics
        else:
            return metrics

    def save(self, file_appendix):
        """
        Save model config
        :param file_appendix: model version (initial, before_update, latest)
        :return: boolean successful
        """
        save_file = self.name + '_' + file_appendix
        save_path = os.path.join(self.save_dir, save_file)
        try:
            self.clf.save(save_path)
        except SaveModelException:
            raise

    def load(self, file_appendix):
        """
        load model config
        :param file_appendix: model version (initial, before_update, latest)
        :return: boolean successful
        """
        load_file = self.name + '_' + file_appendix
        load_path = os.path.join(self.save_dir, load_file)
        try:
            self.clf.load(load_path)
        except LoadModelException:
            raise

    def check_load(self, file_appendix):
        load_file = self.name + '_' + file_appendix
        load_path = os.path.join(self.save_dir, load_file)
        try:
            self.clf.check_load(load_path)
        except LoadModelException:
            raise
