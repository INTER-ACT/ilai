from random import randint

from api.models import DataSet, DataElement, Tag

tags = [
    {'name': 'PRO', 'service': 'sentiment'},
    {'name': 'CONTRA', 'service': 'sentiment'}
]


def get_dataset_1():
    return {'id': 10,
            'name': 'sentiment_data_set',
            'service': 'sentiment',
            'data': get_dataelements()}


def get_dataset_2():
    return {'id': 33,
            'name': 'sentiment_data_set_1',
            'service': 'sentiment',
            'data': get_dataelements()}


def get_dataset_3():
    return {'id': 4,
            'name': 'sentiment_data_set_2',
            'service': 'sentiment',
            'data': get_dataelements()}


def get_tags():
    return tags


def get_dataelements():
    return [
        {'text': 'Lorem ipsum Dolor sit amet', 'tags': ['PRO']},
        {'text': 'Tolle Idee!', 'tags': ['PRO']},
        {'text': 'Idiotisch, wir sollten damit sofort aufhören.', 'tags': ['CONTRA']}
    ]


def get_new_dataelement():
    return {'text': 'New Dataelement', 'tags': ['CONTRA']}
