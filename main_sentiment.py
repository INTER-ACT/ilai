import os.path as path
import numpy as np
import pandas as pd
import random
import time

from services.sentiment.models import LSTMPaddedModel, LSTMSequenceModel, CNNModel, DNNModel, SVMModel, CombinedModel
from services.shared_ressources import init_all_ressources
from exceptions.model_exceptions import LoadModelException
from config import Config

corpus_path = 'data/training_data/trainingsdaten_sentiment_all.csv'
seperation_factor = 0.9
texts_qualitative = [
    "",
    "Das ist toll!",
    "Gute Ergänzung! Ich würde noch vorschlagen §7 miteinzubeziehen.",
    "Ich kann diese Idee aufgrund Paragraph 734 nicht untersützen.",
    "Ich kann diese Idee aufgrund §734 nicht untersützen.",
    "Wie wollt ihr das umsetzen? Für mich erscheint dies Blödsinn zu sein",
]
solution_qualitative = [1, 1, 0, 0, 0]


def train_on_shuffeled_training_set(model, X, Y, validation_split=0.2, runs=5, epochs=10):
    """
    Trains the model x times on a newly shuffeled training set
    :param model:
    :param X: list of texts - [string]
    :param Y: list of tags - [string]
    :param validation_split:
    :param epochs:
    :return:
    """
    train_len = int(X.shape[0] * validation_split)
    metrics = []

    model.save('initial')

    for run in range(runs):
        merged = zip(X, Y)
        random.shuffle(merged)
        X, Y = zip(*merged)

        X_train = X[:train_len]
        Y_train = Y[:train_len]
        X_test = X[:train_len]
        Y_test = Y[:train_len]

        model.load('initial')
        model.train(X_train, Y_train)
        metrics.append(model.text(X_test, Y_test))

    for i in range(len(metrics)):
        print('****************************')
        print('Run: {}\nAccuracy: {} -- {}\nf1: {} -- {}\nprecision: {} -- {}\nrecall: {} -- {}\nconfusion-matrix:\n{}'
              .format(i, metrics[i]['accuracy'], metrics[i]['accuracy'].sum() / len(metrics[i]['accuracy']),
                      metrics[i]['f1'], metrics[i]['f1'].sum() / len(metrics[i]['f1']),
                      metrics[i]['precision'], metrics[i]['precision'].sum() / len(metrics[i]['precision']),
                      metrics[i]['recall'], metrics[i]['recall'].sum() / len(metrics[i]['recall']),
                      metrics[i]['confusion_matrix']))

    # TODO: reshape metrics dict to list in list in order to calculate avg
    print('\n\n######### SUMMARY #########')
    print('not implemented')
    # avg = np.average(np.array(metrics), axis=0)
    # print('Accuracy: {}\nf1: {}\nprecision: {}\nrecall: {}'.format(avg[0], avg[1], avg[2], avg[3]))


def read_data(corpus_path):
    df = pd.read_csv(corpus_path, skipinitialspace=True, sep=';', encoding='utf-8', header=None, names=['Tag', 'Text'],
                     engine='python')
    df = df[1:]  # 1 element is always none
    df = df.iloc[np.random.permutation(len(df))]  # shuffle training set
    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    config = Config()
    init_all_ressources()

    data = read_data(corpus_path)
    set_sepeartion = int(len(data) * seperation_factor)

    X_train = data['Text'][:set_sepeartion].tolist()
    X_test = data['Text'][set_sepeartion:].tolist()

    Y_train = data['Tag'][:set_sepeartion].tolist()
    Y_test = data['Tag'][set_sepeartion:].tolist()

    models = [
        LSTMPaddedModel('LSTM_PAD_31', tags=["NEUTRAL", "CONTRA", "PRO"],
                        save_dir=path.join(config.paths['data'], 'model_saves'),
                        hidden_layer_sizes=[100], dropouts=[0.3], activations=['relu'], sequence_length=50),
        LSTMPaddedModel('LSTM_PAD_32', tags=["NEUTRAL", "CONTRA", "PRO"],
                        save_dir=path.join(config.paths['data'], 'model_saves'),
                        hidden_layer_sizes=[100, 100], dropouts=[0.3, 0.3], activations=['relu', 'relu'],
                        sequence_length=50),
        LSTMPaddedModel('LSTM_PAD_33', tags=["NEUTRAL", "CONTRA", "PRO"],
                        save_dir=path.join(config.paths['data'], 'model_saves'),
                        hidden_layer_sizes=[150], dropouts=[0.3], activations=['relu'],
                        sequence_length=50),
    ]

    print('******** Training Set: ********')
    print('Neutral: {} - Contra: {} - Pro: {} - SUM: {}'.format(Y_train.count('NEUTRAL'), Y_train.count('PRO'),
                                                                Y_train.count('CONTRA'), len(Y_train)))
    print('******** Test Set: ********')
    print('Neutral: {} - Contra: {} - Pro: {} - SUM: {}'.format(Y_test.count('NEUTRAL'), Y_test.count('PRO'),
                                                                Y_test.count('CONTRA'), len(Y_test)))

    results = {}
    for model in models:
        try:
            model.load('latest')
            print('{} loaded'.format(model.name))
        except LoadModelException:
            print('loading failed')
            print('Training {}...'.format(model.name))
            start_time = time.time()
            model.train(X_train, Y_train)
            print('+++++ Training Time: {} +++++'.format(time.time() - start_time))
            model.save('latest')
        results[model.name + '_test'] = model.test(X_test, Y_test)
        results[model.name + '_train'] = model.test(X_train, Y_train)

    for key, result in results.items():
        print('****************************')
        print('ID: {}\nAccuracy: {} -- {}\nf1: {} -- {}\nprecision: {} -- {}\nrecall: {} -- {}\nconfusion-matrix:\n{}'
              .format(key, result['accuracy'], result['accuracy'].sum() / len(result['accuracy']),
                      result['f1'], result['f1'].sum() / len(result['f1']),
                      result['precision'], result['precision'].sum() / len(result['precision']),
                      result['recall'], result['recall'].sum() / len(result['recall']),
                      result['confusion_matrix']))


    print('-*-*-*-*-*-*-*-*-*- QUALITATIVE EVALUATION -*-*-*-*-*-*-*-*-*-')

    for text, solution in zip(texts_qualitative, solution_qualitative):
        print('-----------')
        print(text)
        for model in models:
            print('{} - {} -- {}'.format(model.name, model.predict([text]), solution))

    while True:
        text = input('Text:')
        for model in models:
            print('{} - {}'.format(model.name, model.predict([text])))
