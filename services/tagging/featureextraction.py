import re
import numpy as np

from services.shared_ressources import get_word_vectors


def getEmbeddings():
    return get_word_vectors()

#link_count
def link_vec(text):
    text = text.lower()
    linkvalues = ['http','www']
    vec = np.zeros([1,1])
    words = text.split()

    return len([word for word in words if any(['http' in word, 'www' in word])])
    #print(vec)


#number_count
def number_count(text):
     text = re.sub('[^0-9]+', ' ', text)
     words = text.split()
     return words


def get_Wordvec(word):
    return get_word_vectors()[word]

def get_vecs(text):
    vecs = []
    words = text.split()
    for word in words:
        vecs.append(get_word_vectors()[word])
    return vecs


def get_vecsForTexts(texts):
    vecs4text=[]
    for idx,text in enumerate(texts):
        vec_list = []

        words = text.split()
        for word in words:
            vec_list.append(get_word_vectors()[word])
        vecs4text.append(vec_list)
    return vecs4text