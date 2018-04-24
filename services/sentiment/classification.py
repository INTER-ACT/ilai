import os.path

import keras
import keras.backend as KBackend
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.externals import joblib

from exceptions.model_exceptions import SaveModelException, LoadModelException
from services.abstract_models import AClassifier


class LinearSVMClassifier(AClassifier):
    def __init__(self, c=1, max_iter=1000):
        self.file_extension = '.pkl'
        self.clf = svm.LinearSVC(C=c, max_iter=max_iter, random_state=1)

    def train(self, X, Y):
        self.clf.fit(X, Y)

    def predict(self, X):
        return self.clf.predict(X)

    def save(self, path):
        path += self.file_extension
        try:
            joblib.dump(self.clf, path)
        except BaseException as e:
            print('saving model failed')
            raise SaveModelException('Saving model to {} failed'.format(path)) from e

    def load(self, path):
        path += self.file_extension
        if not os.path.isfile(path):
            raise LoadModelException("Loading model failed. File {} not found.".format(path))
        try:
            self.clf = joblib.load(path)
        except OSError as e:
            raise LoadModelException('Loading model from {} failed'.format(path)) from e


class SVMKernelClassifier(AClassifier):
    def __init__(self, log_dir='./tb', c=1, max_iter=200, kernel='rbf', degree=3):
        self.file_extension = '.pkl'
        self.clf = svm.SVC(C=c, kernel=kernel, gamma='auto', degree=degree, max_iter=max_iter, random_state=1)

    def train(self, X, Y):
        self.clf.fit(X, Y)

    def predict(self, X):
        return self.clf.predict(X)

    def save(self, path):
        path += self.file_extension
        try:
            joblib.dump(self.clf, path)
        except BaseException as e:
            print('saving model failed')
            raise SaveModelException('Saving model to {} failed'.format(path)) from e

    def load(self, path):
        path += self.file_extension
        if not os.path.isfile(path):
            raise LoadModelException("Loading model failed. File {} not found.".format(path))
        try:
            self.clf = joblib.load(path)
        except OSError as e:
            raise LoadModelException('Loading model from {} failed'.format(path)) from e


class DNN(AClassifier):
    def __init__(self, log_dir='./tb', input_size=20, output_size=3, hidden_layer_sizes=[100, 100],
                 activations=['relu', 'relu', 'softmax'], optimizer='adam', loss='categorical_crossentropy',
                 tf_graph=None):
        if len(activations) != len(hidden_layer_sizes) + 1:
            raise ValueError('Layer configurations do not have equal counts of layers')

        self.file_extension = '.h5'
        self.id = log_dir
        self.tf_graph = tf_graph
        if self.tf_graph is None:
            self.tf_graph = KBackend.get_session().graph

        with self.tf_graph.as_default():
            self.tbCallback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True)
            self.model = keras.models.Sequential()
            layer_sizes = [input_size] + hidden_layer_sizes + [output_size]

            for i in range(1, len(layer_sizes)):
                self.model.add(
                    keras.layers.Dense(layer_sizes[i], activation=activations[i - 1], input_dim=layer_sizes[i - 1],
                                       use_bias=True, activity_regularizer=keras.regularizers.l1_l2()))

            self.model.compile(optimizer=optimizer,
                               loss=loss,
                               metrics=['accuracy'])

            # print(self.model.summary())

    def train(self, X, Y, batch_size=32, epochs=10, validation_split=0.1):
        with self.tf_graph.as_default():
            self.model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                           callbacks=[self.tbCallback], verbose=0)

    def predict(self, X):
        with self.tf_graph.as_default():
            return self.model.predict(X, batch_size=1)

    def save(self, path):
        path += self.file_extension
        with self.tf_graph.as_default():
            try:
                self.model.save(path)
            except BaseException as e:
                print('saving model failed')
                raise SaveModelException('Saving model to {} failed'.format(path)) from e

    def load(self, path):
        path += self.file_extension
        if not os.path.isfile(path):
            raise LoadModelException("Loading model failed. File {} not found.".format(path))
        with self.tf_graph.as_default():
            try:
                self.model = keras.models.load_model(path)
            except OSError as e:
                raise LoadModelException('Loading model from {} failed'.format(path)) from e


class LSTMSequence(AClassifier):
    def __init__(self, log_dir='./tb', input_size=300, output_size=3, hidden_layer_sizes=[100], dropouts=[0.0],
                 activations=['relu'], optimizer='adam', loss='categorical_crossentropy', tf_graph=None):
        if len(activations) != len(hidden_layer_sizes) != len(dropouts):
            raise ValueError('Layer configurations do not have equal counts of layers')

        self.file_extension = '.h5'
        self.id = log_dir
        self.tf_graph = tf_graph
        self.output_size = output_size
        if self.tf_graph is None:
            self.tf_graph = KBackend.get_session().graph

        with self.tf_graph.as_default():
            self.tbCallback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True)
            self.model = keras.models.Sequential()
            layer_sizes = [input_size] + hidden_layer_sizes

            for i in range(1, len(layer_sizes)):
                return_sequence = i != len(layer_sizes) - 1
                self.model.add(keras.layers.LSTM(layer_sizes[i], input_shape=(None, layer_sizes[i - 1]), batch_size=1,
                                                 return_sequences=return_sequence, activation=activations[i - 1],
                                                 dropout=dropouts[i - 1]))

            self.model.add(keras.layers.Dense(output_size, activation='softmax'))
            self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

            self.initial_weights = self.model.get_weights()
            # print(self.model.summary())

    def train(self, X, Y, epochs=10, validation_split=0.1, plot_data=False, print_information=False):
        train_len = int(len(X) * validation_split)
        X_train = X[:train_len]
        Y_train = Y[:train_len]
        X_val = X[train_len:]
        Y_val = Y[train_len:]

        train_losses = []
        train_acc = []
        test_losses = []
        test_acc = []

        with self.tf_graph.as_default():
            for epoch in range(epochs):
                epoch_train_loss = 0
                epoch_train_acc = 0
                epoch_test_loss = 0
                epoch_test_acc = 0
                for i in range(len(X_train)):
                    loss, acc = self.model.train_on_batch(X_train[i], Y_train[i].reshape(1, self.output_size))
                    epoch_train_loss += loss
                    epoch_train_acc += acc

                # validation
                for i in range(len(X_val)):
                    loss, acc = self.model.test_on_batch(X_val[i], Y_val[i].reshape(1, self.output_size))
                    epoch_test_loss += loss
                    epoch_test_acc += acc

                train_losses.append(epoch_train_loss / len(X_train))
                train_acc.append(epoch_train_acc / len(X_train))
                test_losses.append(epoch_test_loss / len(X_val))
                test_acc.append(epoch_test_acc / len(X_val))

                if print_information or epoch == epochs - 1:
                    print('----------------')
                    print('epoch #{}\ntrain: loss: {:f} acc: {:f}\ntest: loss: {:f} acc: {:f}'
                          .format(epoch, train_losses[-1], train_acc[-1], test_losses[-1], test_acc[-1]))

        if plot_data:
            plt.plot(train_losses, 'b-', test_losses, 'g-')
            plt.ylabel('loss')
            plt.xlabel('epochs')
            plt.show()

            plt.plot(train_acc, 'b-', test_acc, 'g-')
            plt.ylabel('acc')
            plt.xlabel('epochs')
            plt.show()

    def predict(self, X):
        predictions = []
        with self.tf_graph.as_default():
            for i in range(len(X)):
                predictions.append(self.model.predict_on_batch(X[i]))
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


class LSTMPadded(AClassifier):
    def __init__(self, log_dir='./tb', sequence_length=25, input_size=300, output_size=3, hidden_layer_sizes=[100],
                 dropouts=[0.3], activations=['relu'], embedding_layer=None, gaussian_noise_stddev=.5, optimizer='adam',
                 loss='categorical_crossentropy', tf_graph=None):
        if len(activations) != len(hidden_layer_sizes) != len(dropouts):
            raise ValueError('Layer configurations do not have equal counts of layers')
        if embedding_layer is None:
            raise ValueError('Initialized Embedding Layer must be passed')

        self.file_extension = '.h5'
        self.id = log_dir
        self.tf_graph = tf_graph
        if self.tf_graph is None:
            self.tf_graph = KBackend.get_session().graph

        with self.tf_graph.as_default():
            self.tbCallback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True)
            self.model = keras.models.Sequential()
            layer_sizes = [input_size] + hidden_layer_sizes

            self.model.add(embedding_layer)
            self.model.add(keras.layers.GaussianNoise(gaussian_noise_stddev))

            for i in range(1, len(layer_sizes)):
                return_sequence = i != len(layer_sizes) - 1
                self.model.add(keras.layers.LSTM(layer_sizes[i], input_shape=(sequence_length, layer_sizes[i - 1]),
                                                 return_sequences=return_sequence, activation=activations[i - 1]))
                self.model.add(keras.layers.Dropout(dropouts[i - 1]))

            self.model.add(keras.layers.Dense(output_size, activation='softmax'))
            self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

            # print(self.model.summary())

    def train(self, X, Y, epochs=10, batch_size=10, validation_split=0.1):
        with self.tf_graph.as_default():
            self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
                           callbacks=[self.tbCallback], verbose=0)

    def predict(self, X):
        with self.tf_graph.as_default():
            return self.model.predict(X, batch_size=1)

    def save(self, path):
        path += self.file_extension
        with self.tf_graph.as_default():
            try:
                self.model.save(path)
            except BaseException as e:
                print('saving model failed')
                raise SaveModelException('Saving model to {} failed'.format(path)) from e

    def load(self, path):
        path += self.file_extension
        if not os.path.isfile(path):
            raise LoadModelException("Loading model failed. File {} not found.".format(path))
        with self.tf_graph.as_default():
            try:
                self.model = keras.models.load_model(path)
            except OSError as e:
                raise LoadModelException('Loading model from {} failed'.format(path)) from e


class CNN(AClassifier):
    def __init__(self, log_dir='./tb', embedding_size=300, sequence_length=25, optimizer='adam',
                 loss='categorical_crossentropy', embedding_layer=None, filter_sizes=[[3, 4, 5]], num_filters=[128],
                 paddings=['same'], conv_activations=['relu'], dense_layer_sizes=[128, 3],
                 dense_activations=['relu', 'softmax'], dropout_rate=0.5, tf_graph=None):
        if len(filter_sizes) != len(num_filters) != len(paddings) != len(conv_activations):
            raise ValueError('Conv Layer configurations do not have equal counts of layers')
        if len(dense_layer_sizes) != len(dense_activations):
            raise ValueError('Dense Layer configurations do not have equal counts of layers')
        if embedding_layer is None:
            raise ValueError('Initialized Embedding Layer must be passed')

        self.file_extension = '.h5'
        self.id = log_dir
        self.tf_graph = tf_graph
        if self.tf_graph is None:
            self.tf_graph = KBackend.get_session().graph

        with self.tf_graph.as_default():
            self.tbCallback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True)

            sequence_input = keras.layers.Input(shape=(sequence_length,), dtype='int32')
            embedded_sequences = embedding_layer(sequence_input)
            reshaped = keras.layers.Reshape((sequence_length, embedding_size, 1))(embedded_sequences)

            layers = [reshaped]

            for conv_layer in range(len(filter_sizes)):
                convs = []
                for fsz in filter_sizes[conv_layer]:
                    l_conv = keras.layers.Conv2D(filters=num_filters[conv_layer], kernel_size=(fsz, embedding_size),
                                                 activation=conv_activations[conv_layer], padding=paddings[conv_layer])(
                        layers[-1])

                    if paddings[conv_layer] == 'same':
                        l_pool = keras.layers.MaxPooling2D(pool_size=(int(l_conv.shape[1]), 1), strides=(1, 1),
                                                           padding='valid')(
                            l_conv)  # l_conv.shape[1] == sequence_length after convolution (if padding != 'same')
                    else:  # padding = 'valid'
                        l_pool = keras.layers.MaxPooling2D(pool_size=(l_conv.shape[1] - fsz + 1, 1), strides=(1, 1),
                                                           padding='valid')(
                            l_conv)  # pooling if conv before is padded 'valid'
                    convs.append(l_pool)
                if len(convs) > 1:
                    layers.append(keras.layers.Concatenate(axis=1)(convs))
                else:
                    layers.append(convs[0])

            flattened = keras.layers.Flatten()(layers[-1])
            layers.append(keras.layers.Dropout(dropout_rate, seed=27)(flattened))  # seed is fixed for replicateability

            for layer_size, activation in zip(dense_layer_sizes, dense_activations):
                layers.append(keras.layers.Dense(layer_size, activation=activation)(layers[-1]))

            self.model = keras.models.Model(sequence_input, layers[-1])
            self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

            # print(self.model.summary())

    def train(self, X, Y, epochs=4, batch_size=10, validation_split=0.1):
        with self.tf_graph.as_default():
            self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
                           callbacks=[self.tbCallback], verbose=0)

    def predict(self, X):
        with self.tf_graph.as_default():
            return self.model.predict(X)

    def save(self, path):
        path += self.file_extension
        with self.tf_graph.as_default():
            try:
                self.model.save(path)
            except BaseException as e:
                print('saving model failed')
                raise SaveModelException('Saving model to {} failed'.format(path)) from e

    def load(self, path):
        path += self.file_extension
        if not os.path.isfile(path):
            raise LoadModelException("Loading model failed. File {} not found.".format(path))
        with self.tf_graph.as_default():
            try:
                print('loading model')
                self.model = keras.models.load_model(path)
                print('finished - loading model')
            except OSError as e:
                raise LoadModelException('Loading model from {} failed'.format(path)) from e
