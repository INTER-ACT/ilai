import numpy as np
import pandas as pd

from services.tagging.models import LSTMOneHot
from exceptions.model_exceptions import LoadModelException, SaveModelException, RunTimeException

from config import Config
from services.shared_ressources import init_all_ressources,taglist
from services.tagging import  preprocessing
from services.train_utility import get_solution_vec, calculate_metrics


def getData(path='./data/training_data/trainingsdatei_tagging.csv'):
    df = pd.read_csv(path, skipinitialspace=True, sep=';', encoding='utf-8')
    df_shuffeled = df.iloc[np.random.permutation(len(df))].reset_index(drop=True)

    X = []
    Y = []

    for index, row in df_shuffeled.iterrows():
        tag = str(row['Tag'])
        if tag in taglist:
            X.append(str(row['Kommentar']))
            Y.append(tag)
    return X, Y


def main():
    config = Config()
    init_all_ressources()

    X, Y = getData()
    print(len(X))
    print(len(Y))
    split = 0.2
    train_test_split = int(len(X) * split)

    X_train = X[train_test_split:]
    Y_train = Y[train_test_split:]
    X_test = X[:train_test_split]
    Y_test = Y[:train_test_split]

    #X_train = X[10:20]
    #Y_train = Y[10:20]
    #X_test = X[:10]
    #Y_test = Y[:10]

    #model = LSTMOneHot('tagging', tags=taglist, save_dir=config.paths['model_saves'])

    models = [
        LSTMOneHot('tagging1',tags=taglist,save_dir=config.paths['model_saves'],layer_sizes=[32,64],activation=['tanh','tanh'],optimizer='adam',loss='mean_squared_error',dropout_rate=[0.05,0.05]),
        LSTMOneHot('tagging2',tags=taglist,save_dir=config.paths['model_saves'],layer_sizes=[64,32],activation=['tanh','tanh'],optimizer='adam',loss='mean_squared_error',dropout_rate=[0.05,0.05]),
        LSTMOneHot('tagging3',tags=taglist,save_dir=config.paths['model_saves'],layer_sizes=[32,16],activation=['tanh','tanh'],optimizer='adam',loss='mean_squared_error',dropout_rate=[0.05,0.05]),
        LSTMOneHot('tagging4',tags=taglist,save_dir=config.paths['model_saves'],layer_sizes=[32,32], activation=['tanh', 'tanh'],optimizer='adam',loss='mean_squared_error',dropout_rate=[0.05, 0.05]),
        LSTMOneHot('tagging5',tags=taglist,save_dir=config.paths['model_saves'],layer_sizes=[64,40],activation=['tanh','tanh'],optimizer='adam',loss='mean_squared_error',dropout_rate=[0.05,0.05]),
        LSTMOneHot('tagging6',tags=taglist,save_dir=config.paths['model_saves'],layer_sizes=[40,20],activation=['tanh','tanh'],optimizer='adam',loss='mean_squared_error',dropout_rate=[0.05,0.05]),
        LSTMOneHot('tagging7',tags=taglist,save_dir=config.paths['model_saves'],layer_sizes=[32,32,32],activation=['tanh','tanh','tanh'],optimizer='adam',loss='mean_squared_error',dropout_rate=[0.05,0.05,0.05])
    ]

    results = {}
    for model in models:
        try:
            model.train(X_train,Y_train)
        except RunTimeException as RTE:
            print("[MAIN] {}".format(RTE.args[0]))

        try:
            model.save('latest')
        except SaveModelException as SaveExc:
             print("[MAIN] {}".format(SaveExc.args[0]))

        results[model.name] = model.test(X_test, Y_test)

    for key, result in results.items():
        print('****************************')
        print('ID: {}\nAccuracy: {} -- {}\nf1: {} -- {}\nprecision: {} -- {}\nrecall: {} -- {}\nconfusion-matrix:\n{}'
              .format(key, result['accuracy'], result['accuracy'].sum() / len(result['accuracy']),
                      result['f1'], result['f1'].sum() / len(result['f1']),
                      result['precision'], result['precision'].sum() / len(result['precision']),
                      result['recall'], result['recall'].sum() / len(result['recall']),
                      result['confusion_matrix']))

    # OLD
   # try:
   #     model.train(X_train, Y_train)
   # except RunTimeException as RTE:
   #     print("[MAIN] {}".format(RTE.args[0]))
   #
   #
   # try:
   #     model.save('latest')
   # except SaveModelException as SaveExc:
   #     print("[MAIN] {}".format(SaveExc.args[0]))
   #
   # try:
   #     print("[MAIN] Loading existing Model")
   #     model.load('latest')
   # except LoadModelException as LoadExc:
   #     print(f"[MAIN] {LoadExc.args[0]}")
#
 #   print('[MAIN] Evaluating Metrics')
  #  metrics = model.test(X_test, Y_test)
   # for metric in metrics:
    #    print(
     #       "Category: {}    Accuracy: {}   F1: {}    Precision: {}   Recall: {}".format(metric['category'],
      #                                                                                   metric['accuracy'],
       #                                                                                  metric['f1'],
        #                                                                                 metric['precision'],
         #                                                                                metric['recall']))


main()
