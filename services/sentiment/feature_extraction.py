import re
import numpy as np
from textblob_de import TextBlobDE as TextBlob

from services.shared_ressources import get_word_vectors, sepl, polarity_clues, subjectivity_clues

""" --------- HANDPICKED FEATURESET --------- """


def get_caps_words_count(text):
    blob = TextBlob(text)
    regex = re.compile(r'([A-Z][^a-z])')
    feature = len([w for w in blob.words if regex.search(w)])
    return np.array([[feature]])


def get_punctuation_vector(text):
    punctuations = ['?', '!', '.', '-', ',']
    vec = np.zeros([1, len(punctuations)])

    for char in text:
        if char in punctuations:
            vec[0, punctuations.index(char)] += 1

    return np.array(vec)


def get_last_char_vector(text):
    last_chars = ['.', '!', '?']
    vec = np.zeros([1, len(last_chars) + 1])

    if len(text) > 0 and text[-1] in last_chars:
        vec[0, last_chars.index(text[-1])] = 1
    else:
        vec[0, len(last_chars)] = 1

    return np.array(vec)


def get_sentiment_phrase_score(text):
    score = 0
    blob = TextBlob(text)

    for n in range(1, 4):
        grams = blob.ngrams(n)
        for gram in grams:
            gram_string = ' '.join(gram._collection)
            if gram_string in sepl.keys():
                score += sepl[gram_string]

    return np.array([[score]])


def get_polarity_score(text):
    feature = __get_clues_score(text, polarity_clues)
    return np.array([[feature]])


def get_subjectivity_score(text):
    feature = __get_clues_score(text, subjectivity_clues)
    return np.array([[feature]])


def __get_clues_score(text, clue_dict):
    score = 0
    blob = TextBlob(text)

    for word in blob.words:
        if word in clue_dict.keys():
            score += clue_dict[word]
    return score


""" --------- SEQUENCE FEATURESET --------- """


def get_word_vec_repr(words):
    if len(words) > 0:
        wv_matrix = np.zeros([len(words), get_word_vectors().vector_size])
        for i in range(len(words)):
            wv_matrix[i, :] = get_word_vectors().word_vec(words[i])
        return wv_matrix
    else:
        return np.zeros([1, get_word_vectors().vector_size])


# word vector similarity
positive_words = ['super', 'schoen', 'ja', 'toll']
negative_words = ['schlecht', 'nein', 'unnoetig']
keras_embedding = None


def get_most_similar_words(words):
    return get_word_vectors().similar_by_vector(get_sum_vec(words))


def get_sum_vec(words):
    sum_vec = np.zeros(get_word_vectors().vector_size, dtype=float)

    for word in words:
        wv = get_word_vectors().word_vec(word)
        sum_vec = np.add(sum_vec, wv)

    return np.divide(sum_vec, len(words))


def get_positive_word_vec_similarity(words):
    feature = __get_word_vec_similarity(words, positive_words)
    return np.array([feature])


def get_negative_word_vec_similarity(words):
    feature = __get_word_vec_similarity(words, negative_words)
    return np.array([feature])


def __get_word_vec_similarity(words, similarity_words):
    similarity_scores = []
    for sw in similarity_words:
        score = 0
        for word in words:
            score += get_word_vectors().similarity(sw, word)
        similarity_scores.append(score / len(words))
    return similarity_scores


def get_wv_keras_embedding():
    return get_word_vectors().get_keras_embedding()
    """
    global keras_embedding
    if keras_embedding is None:
        keras_embedding = get_word_vectors().get_keras_embedding()
    return keras_embedding
    """


def get_embedding_indices(words):
    return [get_word_vectors().vocab[w].index for w in words]
