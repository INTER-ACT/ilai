import numpy as np
from keras.preprocessing.sequence import pad_sequences
from textblob_de import TextBlobDE as TextBlob

import services.sentiment.feature_extraction as feature_extraction
import services.sentiment.preprocessing as preprocessing
from services.shared_ressources import get_word_vectors


def pipe_features_before_preprocessing(text):
    feature_vec = feature_extraction.get_punctuation_vector(text)
    feature_vec = np.concatenate((feature_vec, feature_extraction.get_caps_words_count(text)), axis=1)
    feature_vec = np.concatenate((feature_vec, feature_extraction.get_last_char_vector(text)), axis=1)

    return feature_vec


def pipe_features_after_preprocessing(text):
    if len(text) > 0:           # after preprocessing the text could have 0 words left
        blob = TextBlob(text)
        feature_vec = feature_extraction.get_sentiment_phrase_score(text)
        feature_vec = np.concatenate((feature_vec, feature_extraction.get_positive_word_vec_similarity(blob.words)), axis=1)
        feature_vec = np.concatenate((feature_vec, feature_extraction.get_negative_word_vec_similarity(blob.words)), axis=1)
        feature_vec = np.concatenate((feature_vec, feature_extraction.get_polarity_score(text)), axis=1)
        feature_vec = np.concatenate((feature_vec, feature_extraction.get_subjectivity_score(text)), axis=1)

        return feature_vec

    return np.zeros([1, 10])


def pipe_preprocessing(text):
    text = preprocessing.replace_umlauts(text)
    text = preprocessing.remove_numbers(text)
    text = preprocessing.remove_special_chars(text)
    text = preprocessing.remove_punctuation(text)
    text = preprocessing.remove_stop_words(text)
    # text = preprocessing.lemmatize(text)
    text = preprocessing.remove_non_vocab_words(text, get_word_vectors().vocab)

    return text


def get_feature_vec(texts):
    feature_vectors = []

    for text in texts:
        feature_vec = pipe_features_before_preprocessing(text)
        text = pipe_preprocessing(text)
        feature_vec = np.concatenate((feature_vec, pipe_features_after_preprocessing(text)), axis=1)
        feature_vectors.append(feature_vec)

    return np.reshape(np.array(feature_vectors), [len(feature_vectors), feature_vectors[0].shape[1]])


def get_wv_vec(texts, max_dimension=6000):
    texts_wv = []
    for text in texts:
        text = pipe_preprocessing(text)
        blob = TextBlob(text)
        wv = feature_extraction.get_word_vec_repr(blob.words)
        wv = wv.reshape([1, wv.shape[1] * wv.shape[0]])
        if wv.shape[1] > max_dimension:
            wv = np.reshape(wv[0, :max_dimension], (1, max_dimension))
        elif wv.shape[1] < max_dimension:
            wv = np.pad(wv, ((0, 0), (0, max_dimension - wv.shape[1])), 'constant', constant_values=0)

        texts_wv.append(wv)

    return np.reshape(np.array(texts_wv), [len(texts_wv), max_dimension])


def get_wv_vec_sequence(texts):
    texts_wv = []
    for text in texts:
        text = pipe_preprocessing(text)
        blob = TextBlob(text)
        wv = feature_extraction.get_word_vec_repr(blob.words)
        texts_wv.append(wv.reshape((1, wv.shape[0], wv.shape[1])))

    return texts_wv


def get_embedding_indices(texts, maxlen=None):
    texts_indices = []
    for text in texts:
        text = pipe_preprocessing(text)
        blob = TextBlob(text)
        indices = feature_extraction.get_embedding_indices(blob.words)
        texts_indices.append(indices)

    if maxlen:
        return pad_sequences(texts_indices, maxlen=maxlen, padding='post', truncating='post')
    return texts_indices
