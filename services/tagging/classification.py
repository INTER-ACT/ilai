import os.path

from abc import ABCMeta, abstractmethod
import numpy as np
import keras.backend as KBackend
from keras.layers import GaussianNoise
from keras.layers import LSTM, Dropout, Dense
from keras.models import Sequential
from keras.models import model_from_json
import keras

from matplotlib import pyplot

import services.tagging.preprocessing as preprocessing
from services.abstract_models import AClassifier
from exceptions.model_exceptions import SaveModelException, LoadModelException, RunTimeException


class LSTMOneHotLabels(AClassifier):


    def __init__(self, model_name, tf_graph=None,input_size=300,dropout_rate = [0.2,0.2],layer_sizes=[100,100],activation = ['tanh','tanh'],optimizer = 'adam',loss = 'mean_squared_error'):
        self.inputSize = input_size
        self.model_name = model_name
        self.file_extension = '.h5'
        self.tf_graph = tf_graph
        if self.tf_graph is None:
            self.tf_graph = KBackend.get_session().graph

        with self.tf_graph.as_default():
            self.model = Sequential()
            self.model.add(GaussianNoise(0.1, input_shape=(None, self.inputSize)))

            for idx in range(len(layer_sizes)-1):
                self.model.add(LSTM(layer_sizes[idx], input_shape=(None, self.inputSize), return_sequences=True, batch_size=1, activation=activation[idx]))
                self.model.add(Dropout(dropout_rate[idx]))

            self.model.add(LSTM(layer_sizes[-1], input_shape=(None, self.inputSize), return_sequences=False, batch_size=1, activation=activation[-1]))
            self.model.add(Dropout(dropout_rate[-1]))
            self.model.add(Dense(10, activation='softmax'))

            self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
            # print(self.model.summary())

    def train(self, X, Y, epochs=25, split=0.10, print_information=False):

        splitVal = int(len(X) * split)
        X_test = X[:splitVal]
        Y_test = Y[:splitVal]
        X_train = X[splitVal:]
        Y_train = Y[splitVal:]
        train_losses = []
        train_acc = []
        test_losses = []
        test_acc = []
        actual_acc = 0.00;

        try:
             with self.tf_graph.as_default():
                for idx in range(epochs):
                    if actual_acc < 0.60:
                        if print_information:
                            print("Epoch {}".format(idx))
                        epoch_train_loss = 0
                        epoch_train_acc = 0
                        epoch_test_loss = 0
                        epoch_test_acc = 0
                        for i in range(len(X_train)):
                            if print_information:
                                print("Training with text {}".format(i))
                            x_train = np.array(X_train[i])
                            y_train = np.array(Y_train[i])
                            x_train = x_train.reshape(1, len(x_train), self.inputSize)
                            y_train = y_train.reshape(1, 10)
                            loss, acc = self.model.train_on_batch(x_train, y_train)
                            epoch_train_loss += loss
                            epoch_train_acc += acc


                        for i in range(len(X_test)):
                            x_test = np.array(X_test[i])
                            y_test = np.array(Y_test[i])
                            x_test = x_test.reshape(1, len(x_test), self.inputSize)
                            y_test = y_test.reshape(1, 10)

                            loss, acc = self.model.test_on_batch(x_test, y_test)
                            epoch_test_loss += loss
                            epoch_test_acc += acc

                        train_losses.append(epoch_train_loss / len(X_train))
                        train_acc.append(epoch_train_acc / len(X_train))
                        test_losses.append(epoch_test_loss / len(X_test))
                        test_acc.append(epoch_test_acc / len(X_test))

                        if idx == epochs - 1 and print_information:
                            print('----------------')
                            print('epoch #{}\ntrain: loss: {:f} acc: {:f}\ntest: loss: {:f} acc: {:f}'
                                  .format(idx, train_losses[-1], train_acc[-1], test_losses[-1], test_acc[-1]))
                        actual_acc = test_acc[-1]
                        #print("====================")
                        #print("Epoch {}: \nloss_train = {} \nacc_train = {} \nloss_test = {} \nacc_test = {}".format(idx,loss_train,acc_train,loss_test,acc_test))
                        #print("====================")
                    else:
                        break;

                prediction_correct = 0
                test_samples = 0

                # calculate accuracy (test method in MODEL)
                for i in range(0, len(X_test)):
                    if print_information:
                        print("Testing Batch - ", i)
                    x_test = np.array(X_test[i])
                    y_test = np.array(Y_test[i])
                    x_test = x_test.reshape(1, len(x_test), self.inputSize)
                    y_test = y_test.reshape(1, 10)
                    loss_test, acc_test = self.model.test_on_batch(x_test, y_test)
                    test_samples += 1
                    if acc_test == 1.0:
                        prediction_correct += 1

                accuracy = prediction_correct / test_samples
                accuracy = accuracy * 100
                if print_information:
                    print("feeding input --> FINISHED")
                    print("ACCURACY: ", accuracy, "%")
        except BaseException as e:
            print("[CLASSIFICATION] Error occurred during training: " + str(e))
            raise RuntimeError("Error occurred during training")

        if print_information:
            pyplot.plot(train_losses, 'r-', test_losses, 'b-')
            pyplot.xlabel('epoch')
            pyplot.ylabel('loss')
            pyplot.title(self.model_name + ' ' + actual_acc)
            pyplot.savefig('data/training_stats/' + self.model_name + 'losses.png')
            pyplot.close()

            pyplot.plot(train_acc, 'r-', test_acc, 'b-')
            pyplot.xlabel('epoch')
            pyplot.ylabel('acc')
            pyplot.title(self.model_name + ' ' + actual_acc)
            pyplot.savefig('data/training_stats/' + self.model_name + 'acc.png')
            pyplot.close()



    def predict(self, X):
        predictions = []
        with self.tf_graph.as_default():
            for i in range(len(X)):
                x_predict = np.array(X[i])
                x_predict = x_predict.reshape(1, len(x_predict), self.inputSize)
                predictions.append(self.model.predict_on_batch(x_predict))
            return np.reshape(np.array(predictions), (len(predictions), predictions[0].shape[1]))

    def save(self, path):
        path += self.file_extension
        with self.tf_graph.as_default():
            try:
                self.model.save(path)
            except BaseException as e:
                print('Saving model to {} failed'.format(path))
                raise SaveModelException('Saving model to {} failed'.format(path)) from e

    def load(self, path):
        path += self.file_extension
        if not os.path.isfile(path):
            raise LoadModelException("Loading model failed. File {} not found.".format(path))
        with self.tf_graph.as_default():
            try:
                self.model = keras.models.load_model(path)
            except OSError as e:
                print('Loading model from {} failed'.format(path))
                raise LoadModelException('Loading model from {} failed'.format(path)) from e


                        #def save(self, filename):
    #    try:
    #        with self.tf_graph.as_default():
    #            model_json = self.model.to_json()
    #            with open(f"{filename}.json", "w") as json:
    #                json.write(model_json)
    #            self.model.save_weights(f"{filename}.h5")
    #            print("Saving Model --> FINISHED")
    #    except BaseException as e:
    #        print(f'Saving model to {filename}.h5 failed')
    #        raise SaveModelException(f'Saving model to {filename}.h5 failed') from e

    #def load(self, filename):
    #    try:
    #        with self.tf_graph.as_default():
    #            json = open(f'{filename}.json', 'r')
    #            loaded_json = json.read()
    #            json.close()
    #            loaded_model = model_from_json(loaded_json)
    #            loaded_model.load_weights(f"{filename}.h5")
    #            self.model = loaded_model
    #            print("Loading Model --> FINISHED")
    #    except OSError as e:
    #        print(f'Loading model from {filename}.h5 failed')
    #        raise LoadModelException(f'Loading model from {filename}.h5 failed') from e

# def reset(self):
#    self.clf.model.set_weights(self.init_weights)
