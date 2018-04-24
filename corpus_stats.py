from collections import Counter

import numpy as np
import pandas as pd
from textblob_de import TextBlobDE as TextBlob

import services.sentiment.feature_extraction as feature_extraction
import services.sentiment.preprocessing as preprocessing
from services.shared_ressources import init_all_ressources, get_word_vectors
from config import Config

corpus_path = 'data/training_data/fuck-this-shit.csv'

def get_most_significant_words(corpus):
    words = []

    for text in corpus:
        blob = TextBlob(text)
        words += blob.words._collection

    return Counter(words)


def pipe_features_before_preprocessing(corpus):
    punctuation, caps_count, last_char = [], [], []
    for text in corpus:
        blob = TextBlob(text)
        punctuation.append(feature_extraction.get_punctuation_vector(text))
        caps_count.append(feature_extraction.get_caps_words_count(text))
        last_char.append(feature_extraction.get_last_char_vector(text))

    return punctuation, caps_count, last_char


def pipe_features_after_preprocessing(corpus):
    sepl, word_vec, sum_vec, pos_similarity, neg_similarity, polarity_score, subjectivity_score = [], [], [], [], [], \
                                                                                                  [], []
    for text in corpus:
        if len(text) > 0:           # after preprocessing the text could have 0 words left
            blob = TextBlob(text)
            sepl.append(feature_extraction.get_sentiment_phrase_score(text))
            word_vec.append(feature_extraction.get_word_vec_repr(blob.words))
            sum_vec.append(feature_extraction.get_sum_vec(blob.words))
            pos_similarity.append(feature_extraction.get_positive_word_vec_similarity(blob.words))
            neg_similarity.append(feature_extraction.get_negative_word_vec_similarity(blob.words))
            polarity_score.append(feature_extraction.get_polarity_score(text))
            subjectivity_score.append(feature_extraction.get_subjectivity_score(text))
    msw = get_most_significant_words(corpus)
    return sepl, pos_similarity, neg_similarity, polarity_score, word_vec, sum_vec, subjectivity_score, msw


def pipe_preprocessing(corpus):
    for i in range(len(corpus)):
        text = corpus[i]
        text = preprocessing.replace_umlauts(text)
        text = preprocessing.remove_numbers(text)
        text = preprocessing.remove_special_chars(text)
        text = preprocessing.remove_punctuation(text)
        text = preprocessing.remove_stop_words(text)
        text = preprocessing.remove_non_vocab_words(text, get_word_vectors().vocab)
        corpus[i] = text
    return corpus


def print_corpus_statistic(punctuation_feature, caps_count_feature, last_char_feature, sepl_feature,
                           pos_similarity_feature, neg_similarity_feature, polarity_score_feature,
                           subjectivity_score_feature, msw, element_count):
    __print_stat(punctuation_feature, 'PUNCTUATIONS')
    __print_stat(caps_count_feature, 'CAPS COUNT')
    __print_stat(last_char_feature, 'LAST CHAR')
    __print_stat(sepl_feature, 'SEPL')
    __print_stat(pos_similarity_feature, 'POS SIMILARITY')
    __print_stat(neg_similarity_feature, 'NEG SIMILARITY')
    __print_stat(polarity_score_feature, 'POLARITY SCORE')
    __print_stat(subjectivity_score_feature, 'SUBJECTIVITY SCORE')
    print('MOST SIGNIFICANT WORDS: {}'.format(msw.most_common(10)))
    print('NUM ELEMENTS: {}'.format(element_count))


def __print_stat(feature, feature_name):
    np_feature = np.array(feature)
    print('{}: min: {} - max: {} - avg: {}'.format(feature_name.upper(), np_feature.min(axis=0), np_feature.max(axis=0),
          np_feature.mean(axis=0)))


def run(corpus):
    punctuation_feature, caps_count_feature, last_char_feature = pipe_features_before_preprocessing(corpus)
    corpus = pipe_preprocessing(corpus)
    sepl_feature, pos_similarity_feature, neg_similarity_feature, polarity_score_feature, word_vec_feature, \
    sum_vec_feature, subjectivity_score_feature, msw = pipe_features_after_preprocessing(corpus)
    print_corpus_statistic(punctuation_feature, caps_count_feature, last_char_feature, sepl_feature,
                           pos_similarity_feature, neg_similarity_feature, polarity_score_feature,
                           subjectivity_score_feature, msw, len(corpus))


if __name__ == "__main__":
    config = Config()
    init_all_ressources()

    df = pd.read_csv(corpus_path, skipinitialspace=True, sep=';', encoding='utf-8', header=None, names=['Tag', 'Text'],
                     engine='python')
    df = df.iloc[1:]
    tag_list = set(df['Tag'])

    for tag in tag_list:
        corpus = df.loc[df['Tag'] == tag]['Text'].tolist()
        print('\n------------- {} -------------'.format(tag))
        run(corpus)
