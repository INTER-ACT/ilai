import threading
import gc

from api.apps import MODEL_MANIPULATION_SEM
from exceptions.model_exceptions import LoadModelException


class LoadingThread(threading.Thread):

    def __init__(self, thread_name, model, version):
        self.production_model = model
        self.version = version
        super().__init__(name=thread_name)

    def run(self):
        print('Starting Loading Thread...')
        with MODEL_MANIPULATION_SEM:
            print('Initializing and loading model: {}'.format(self.production_model.name))

            load_model = self.production_model.__class__(model_name=self.production_model.name,
                                                         tags=self.production_model.tags,
                                                         save_dir=self.production_model.save_dir)
            try:
                load_model.load(self.version)
                self.production_model = load_model
            except LoadModelException as e:
                print(e)

            del load_model

        gc.collect()
        print('FINISHED Loading Thread')
