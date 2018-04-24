import json
import re
import numpy as np
import pandas as pd
from textblob_de import TextBlobDE as TextBlob

from config import Config
from services.shared_ressources import stop_words, taglist

csv_path = 'services/tagging/data/Tags_Kommentare.csv'


# stopwords
def remove_stopwords(text):
    textwords = text.split()
    resultwords = [item for item in textwords if item not in stop_words]
    text = ' '.join(resultwords)
    return text


# leerzeichen minimieren
def remove_spaces(text):
    return text.replace('  ', ' ')


# umlaute ausschreiben
def replace_umlaute(text):
    # print("[Replace Umlaute]")
    chars = {'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'Ä': 'Ae', 'Ö': 'Oe', 'Ü': 'Ue'}
    for char in chars:
        text = text.replace(char, chars[char])
    return text


# satz-und sonderzeichen entfernen
def remove_satzsonder(text):
    return re.sub('[^0-9a-zA-Z]+', ' ', text)


# lemmatization
def lemmatization(text):
    # print("[Lemmatization]")
    return TextBlob(text).words.lemmatize()._collection


# kleinschreibung notewendig?
def tolower(text):
    return text.lower()


def remove_non_vocab(words, wordvecs):
    resultwords = [word for word in words if word in wordvecs]
    return ' '.join(resultwords)


def preprocessing_pipeline(text, wordvecs):
    text = remove_stopwords(text)
    text = replace_umlaute(text)
    text = remove_satzsonder(text)
    text = lemmatization(text)
    text = remove_non_vocab(text, wordvecs)
    return text


def preprocess_texts(texts, wordvecs):
    config = Config()

    for idx, text in enumerate(texts):
        tmp_text = str(text)
        tmp_text = remove_stopwords(tmp_text)

        tmp_text = replace_umlaute(tmp_text)
        tmp_text = remove_satzsonder(tmp_text)
        tmp_text = lemmatization(tmp_text)
        tmp_text = remove_non_vocab(tmp_text, wordvecs)
        texts[idx] = tmp_text
    return texts


# deprecated
def delete_unecessary():
    df = pd.read_csv(csv_path, skipinitialspace=True, sep=';', encoding='utf-8')
    for idx, row in df.iterrows():
        text = row['Kommentar']
        words = text.split(' ')
        print(len(words))
        if len(words) <= 2:
            df.drop(df.index[idx])


def get_data():
    train_size = 15
    nrows = 20

    df = pd.read_csv(csv_path, skipinitialspace=True, sep=';', encoding='utf-8')
    df_shuffeled = df.iloc[np.random.permutation(len(df))].reset_index(drop=True)

    x_insg = []
    y_insg = []

    for index, row in df_shuffeled.iterrows():
        if index > nrows:
            break
        else:
            tag = str(row['Tag'])
            if tag in taglist:
                x_insg.append(str(row['Kommentar']))
                y_insg.append(tag)
    return train_test_split(x_insg, y_insg, train_size)


def get_data_withOneHot(nrows, value):
    df = pd.read_csv(csv_path, skipinitialspace=True, sep=';', encoding='utf-8')
    df_shuffeled = df.iloc[np.random.permutation(len(df))].reset_index(drop=True)

    y = []
    x = []
    tags = []
    for idx, row in df_shuffeled.iterrows():
        if idx > nrows:
            break
        else:
            tag = str(row['Tag'])
            if tag in taglist:
                x.append(str(row['Kommentar']))
                y.append(np.zeros(len(taglist), dtype=float))
                y[idx][taglist.index(tag)] = 1
                tags.append(str(row['Tag']))
    x_train, y_train, x_test, y_test, tag_train, tag_test = train_test_split(x, y, tags, value)
    return x_train, y_train, x_test, y_test, tag_train, tag_test


def train_test_split(x, y, tags, value):
    x_train = x[:value]
    y_train = y[:value]
    tag_train = tags[:value]
    x_test = x[value + 1:]
    y_test = y[value + 1:]
    tag_test = tags[value + 1:]
    return x_train, y_train, x_test, y_test, tag_train, tag_test


def oneHotEncoding(texts, solution):
    y = []
    for idx, text in enumerate(texts):
        tag = solution[idx]
        if tag in taglist:
            y.append(np.zeros(len(taglist), dtype=float))
            y[idx][taglist.index(tag)] = 1
    y = np.array(y)
    return y
