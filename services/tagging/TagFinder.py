from collections import Counter

import pandas as pd
import services.tagging.preprocessing as preprocessing
from sklearn.utils import shuffle

import services.tagging.featureextraction as featureextraction

taglist = ['User-Generated_Content','Wirtschaftliche Interessen','Download und Streaming','Rechteinhaberschaft','Respekt und Anerkennung','Freiheiten der Nutzer','Bildung und Wissenschaft','Kulturelles Erbe ','soziale Medien','Nutzung fremnder Inhalte']
csv_path = 'data/Tags_Kommentare.csv'
df = pd.read_csv(csv_path, skipinitialspace=True, sep=';', encoding='utf-8')
df = shuffle(df)

def loop_through_comments():
    y_text=[]
    x_text=[]
    global gesamttext
    gesamttext = ''
    for i in taglist:
        gesamttext= ' '
        currtag = str(i)
        print(currtag)
        #geht alle zeilen für diesen Tag durch
        for index,row in df.iterrows():
            tag = str(row['Tag'])
            if tag in currtag:



                text = str(row['Kommentar'])
                y_text.append(tag)
                x_text = str(row['Kommentar'])
                gesamttext = gesamttext+text

        gesamttext = preprocessing.remove_satzsonder(gesamttext)
        gesamttext = preprocessing.remove_stopwords(gesamttext)
        gesamttext = ' '.join(preprocessing.lemmatization(gesamttext))
        gesamttext = preprocessing.remove_non_vocab(gesamttext, featureextraction.wordvecs.vocab)
        print(gesamttext)
        print('wortzahl:')
        unique_words(gesamttext)



def unique_words(text):
    print(Counter(text.lower().split()))


featureextraction.create_wordvec()
loop_through_comments()