from threading import Semaphore

from django.apps import AppConfig

taggingModelController = None
sentimentModelController = None
TRUE_TOKEN = None
MODEL_MANIPULATION_SEM = Semaphore(1)
MODEL_MANIPULATION_THREADS = []


class ApiConfig(AppConfig):
    name = 'api'

    def ready(self):
        global taggingModelController
        global sentimentModelController
        global TRUE_TOKEN

        print('initializing app...')

        from keras import backend as KBackend

        KBackend.clear_session()

        from config import Config
        from services.shared_ressources import init_all_ressources, taglist
        from api.ai_model_controller import AIModelController
        from services.sentiment.models import CNNModel
        from services.tagging.models import LSTMOneHot
        from exceptions.model_exceptions import LoadModelException

        config = Config()
        TRUE_TOKEN = config.api['token']

        print('initializing shared ressources')
        init_all_ressources()

        print('initializing tagging model...')
        tagging_model = LSTMOneHot(model_name='tagging', tags=taglist,
                                   save_dir=config.paths['model_saves'], tf_graph=KBackend.get_session().graph)
        try:
            tagging_model.load('latest')
            print('loaded latest tagging model SUCCESSFULLY')
        except LoadModelException:
            print('loading latest tagging model FAILED')
        taggingModelController = AIModelController(tagging_model, service="tagging")

        print('initializing sentiment model...')
        sentiment_model = CNNModel(model_name='sentiment', tags=["NEUTRAL", "CONTRA", "PRO"],
                                   save_dir=config.paths['model_saves'], tf_graph=KBackend.get_session().graph)
        try:
            sentiment_model.load('latest')
            print('loaded latest sentiment model SUCESSFULLY')
        except LoadModelException:
            print('loading latest sentiment model FAILED')
        sentimentModelController = AIModelController(sentiment_model, service="sentiment")
        print('initializing models - FINISHED')


