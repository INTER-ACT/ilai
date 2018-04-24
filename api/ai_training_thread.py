import threading
import gc

from api.apps import MODEL_MANIPULATION_SEM
from exceptions.model_exceptions import SaveModelException, LoadModelException


class TrainingThread(threading.Thread):

    def __init__(self, thread_name, model, texts, solutions):
        self.production_model = model
        self.texts = texts
        self.solutions = solutions
        super().__init__(name=thread_name)

    def run(self):
        print('Starting Training Thread...')
        with MODEL_MANIPULATION_SEM:
            print('Initializing and loading training model: {}'.format(self.production_model.name))

            train_model = self.production_model.__class__(model_name=self.production_model.name,
                                                          tags=self.production_model.tags,
                                                          save_dir=self.production_model.save_dir)

            try:
                train_model.load('latest')
            except LoadModelException:
                print('no latest model found - training a new model')

            print('saving model copy...')
            try:
                train_model.save('before_training')
            except SaveModelException as e:
                print(e)
                print('Saving before_training model failed')
                print('STOPPING training model: {}'.format(train_model.name))
                return

            print('START training model: {}...'.format(train_model.name))
            train_model.train(self.texts, self.solutions)

            try:
                train_model.save('latest')
            except SaveModelException as e:
                print(e)
                print('trained model NOT saved: {}'.format(train_model.name))

            print('training model: {} - FINISHED'.format(train_model.name))
            production_model = train_model
            print('Loaded trained model: {}'.format(production_model.name))
            print('Training Process - FINISHED')

        del train_model
        gc.collect()
