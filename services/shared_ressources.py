import io
import csv
import json

from gensim.models import KeyedVectors

from config import Config

taglist = []
stop_words = []
_word_vectors = None
sepl = {}
polarity_clues = {}
subjectivity_clues = {}
punctuations = ['.', ',', '"', '!', '?', '-', '_']
umlauts = {
    'ä': 'ae',
    'ö': 'oe',
    'ü': 'ue',
    'Ä': 'Ae',
    'Ö': 'Oe',
    'Ü': 'Ue',
    'ß': 'ss'
}

# needed because _word_vectors reference changes when init is called
def get_word_vectors():
    return _word_vectors


def init_all_ressources():
    config = Config()
    init_taglist(config.paths['taglist'])
    init_stop_words(config.paths['stopwords'])
    init_sepl(config.paths['sepl'])
    init_polarity_clues(config.paths['polarity_clues'])
    init_subjectivity_clues(config.paths['subjectivity_clues'])
    init_word_vectors(config.paths['word_vector_model'])


def init_taglist(taglist_path):
    global taglist
    print('initializing taglist...')
    with io.open(taglist_path, encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        for line in reader:
            taglist.append(line[0])

    print('initializing taglist - FINISHED')


def init_stop_words(stop_words_path):
    global stop_words

    print('initializing stop words...')
    with io.open(stop_words_path, encoding='utf-8') as file:
        stop_words = json.load(file)
    print('initializing stop words - FINISHED')


def init_word_vectors(word_vector_path):
    global _word_vectors
    print('initializing word vectors...')
    _word_vectors = KeyedVectors.load_word2vec_format(word_vector_path, binary=True)
    print('initializing word vectors - FINISHED')


def init_sepl(sepl_path):
    global sepl
    print('initializing sepl...')
    with io.open(sepl_path, encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        for line in reader:
            if line[0][0] != '#':
                sepl[line[0]] = float(line[1])
    print('initializing sepl - FINISHED')


def init_polarity_clues(polarity_path):
    global polarity_clues
    print('initializing polarity clues...')
    with io.open(polarity_path, encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        for line in reader:
            value = 0
            if line[3] == 'negative':
                value = -1
            if line[3] == 'positive':
                value = 1
            polarity_clues[line[1]] = value
    print('initializing polarity clues - FINISHED')


def init_subjectivity_clues(subjectivity_path):
    global subjectivity_clues
    print('initializing subjectivity clues...')
    with io.open(subjectivity_path, encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        for line in reader:
            value = 0
            if line[3] == '1':
                value = 1
            elif line[4] == '1':
                value = -1
            subjectivity_clues[line[1]] = value
    print('initializing subjectivity clues - FINISHED')
