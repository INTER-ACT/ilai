import re
from textblob_de import TextBlobDE as TextBlob

from services.shared_ressources import punctuations, umlauts, stop_words

""" --------- REMOVE ---------"""

def remove_numbers(text):
    return re.sub(r'[0-9]', '', text)


def remove_punctuation(text):
    for c in punctuations:
        text = text.replace(c, '')
    return text


def remove_special_chars(text):
    return re.sub(r'[^A-Za-z0-9äöü]', ' ', text)

def minimizing_space(text):
    elements = [element for element in text.split(' ') if element != '']
    return ' '.join(elements)


def remove_stop_words(text):
    blob = TextBlob(text)
    return ' '.join([w for w in blob.words if w.lower() not in stop_words])


def remove_non_vocab_words(text, vocab):
    blob = TextBlob(text)
    return ' '.join([w for w in blob.words if w in vocab])


""" --------- REPLACE --------- """

def replace_umlauts(text):
    for umlaut, replacement in umlauts.items():
        text = text.replace(umlaut, replacement)
    return text


def lemmatize(text):
    blob = TextBlob(text)
    return ' '.join(blob.words.lemmatize()._collection)