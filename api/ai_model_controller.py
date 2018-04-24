import gc

from api.ai_load_thread import LoadingThread
from api.apps import MODEL_MANIPULATION_THREADS
from exceptions.model_exceptions import LoadModelException, SaveModelException
from api.models import DataSet
from api.ai_training_thread import TrainingThread


class AIModelController:

    def __init__(self, model, service):
        self.model = model
        self.service = service

    def _get_data(self, dataset_ids):
        """
        fetch texts + solutions of datasets from database
        :param dataset_ids: List of dataset ids
        :return: [texts, solutions] solutions = pd.Series[tag, tag]
        """
        try:
            datasets = [DataSet.objects.get(pk=id) for id in dataset_ids]
        except DataSet.DoesNotExist:
            print("Requested dataset does not exist")
            raise

        texts = []
        solutions = []
        for dataset in datasets:
            for dataelement in dataset.data.all():
                if(len(dataelement.tags.all()) > 1):
                    for tag in dataelement.tags.all():
                        texts.append(dataelement.text)
                        solutions.append(tag.name)
                else:
                    solutions.append(dataelement.tags.all()[0].name)
                    texts.append(dataelement.text)

        return texts, solutions

    def predict(self, request_data):
        """
        return predictions of given texts
        :param request_data: dict() includes keys 'texts' and 'threshold'
        :return: List of dict() {text_id, tags}
        """
        threshold = request_data['threshold']
        texts = [t['text'] for t in request_data['texts']]
        text_ids = [t['text_id'] for t in request_data['texts']]

        predictions = self.model.predict(texts, threshold)

        if len(text_ids) != len(predictions):
            print('length of prediction list does not match count of text ids')

        predictions_data = []
        for text_id, prediction in zip(text_ids, predictions):
            if not isinstance(prediction, list) and prediction != []:
                prediction = [prediction]
            predictions_data.append({'text_id': text_id, 'tags': prediction})

        return predictions_data

    def train(self, request_data):
        """
        start a new asynchronous training thread
        :param request_data: dict() includes key 'dataset_ids'
        :return: None
        """
        texts, solutions = self._get_data(request_data['dataset_ids'])
        trainingThread = TrainingThread('Train {}'.format(self.model.name), self.model, texts, solutions)
        trainingThread.start()
        for i in range(10):
            gc.collect()

    def test(self, request_data):
        """
        return test-metrics of AI on given dataset
        :param request_data: dict() includes key 'dataset_ids'
        :return: metrics dict(): accuracy, f1, precision, recall
        """
        texts, solutions = self._get_data(request_data['dataset_ids'])
        metrics = self.model.test(texts, solutions, avg_metrics=True)
        return metrics

    def load(self, request_data):
        """
        loads requested model
        :param request_data: dict() includes key 'model_version'
        :return: bool successful
        """
        model_version = request_data['model_version']
        try:
            self.model.check_load(model_version)
        except LoadModelException as e:
            print("Load Model check failed: " + e)
            raise

        loadThread = LoadingThread('Load {}'.format(self.model.name), self.model, model_version)
        loadThread.start()
        gc.collect()
