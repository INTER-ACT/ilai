import random
import os.path
import pandas as pd


def equal_size():
    corpus_path = 'data/training_data/trainingsdaten_sentiment_stance_all.csv'
    save_path = 'data/training_data/stance_equal_size.csv'

    df = pd.read_csv(corpus_path, skipinitialspace=True, sep=';', encoding='utf-8', header=None, names=['Tag', 'Text'])
    # tag_list = set(df['Tag'])
    tag_list = ['PRO', 'CONTRA']

    tag_corpa = {
        'Tag': [],
        'Text': []
    }

    min_count = min([len(df.loc[df['Tag'] == tag]) for tag in tag_list])

    for tag in tag_list:
        tag_corpus = df.loc[df['Tag'] == tag]['Text'].tolist()
        indices = random.sample(range(0, len(tag_corpus)), min_count)
        tag_corpa['Text'] += df['Text'].iloc[indices].tolist()
        tag_corpa['Tag'] += [tag] * len(indices)

    df_resized = pd.DataFrame(tag_corpa)

    df_resized.to_csv(save_path, sep=';', header=False, index=False)


def combine_corpa():
    path_stance = 'data/training_data/tmp_utf_new.csv'
    path_arg = 'data/training_data/trainingsdaten_sentiment_argument_all.csv'
    save_path = 'data/training_data/trainingsdaten_sentiment_all.csv'

    df_stance = pd.read_csv(path_stance, skipinitialspace=True, sep=';', encoding='utf-8', header=None,
                            names=['Tag', 'Text'])
    df_arg = pd.read_csv(path_arg, skipinitialspace=True, sep=';', encoding='utf-8', header=None, names=['Tag', 'Text'])

    neutral = df_arg.loc[df_arg['Tag'] == 'NON-ARGUMENTATIVE']
    neutral = neutral.replace('NON-ARGUMENTATIVE', 'NEUTRAL')

    combinded = pd.concat([df_stance, neutral])
    print(combinded)

    combinded.to_csv(save_path, sep=';', header=False, index=False)


def combine_corpa():
    path_stance = 'data/training_data/tmp_utf_new.csv'
    path_arg = 'data/training_data/trainingsdaten_sentiment_argument_all_utf.csv'
    save_path = 'data/training_data/trainingsdaten_sentiment_all_tmp.csv'

    df_stance = pd.read_csv(path_stance, skipinitialspace=True, sep=';', encoding='utf-8', header=None,
                            names=['Tag', 'Text'])
    df_arg = pd.read_csv(path_arg, skipinitialspace=True, sep=';', encoding='utf-8', header=None, names=['Tag', 'Text'])

    neutral = df_arg.loc[df_arg['Tag'] == 'NON-ARGUMENTATIVE']
    neutral = neutral.replace('NON-ARGUMENTATIVE', 'NEUTRAL')

    combinded = pd.concat([df_stance, neutral])
    print(combinded)

    combinded.to_csv(save_path, sep=';', header=False, index=False)


if __name__ == '__main__':
    combine_corpa()
