from random import randint

from api.models import DataSet, DataElement, Tag

tags = [
    {'name': 'User-Generated_Content', 'service': 'tagging'},
    {'name': 'Wirtschaftliche Interessen', 'service': 'tagging'},
    {'name': 'Download und Streaming', 'service': 'tagging'},
    {'name': 'Rechteinhaberschaft', 'service': 'tagging'},
    {'name': 'Respekt und Anerkennung', 'service': 'tagging'},
    {'name': 'Freiheiten der Nutzer', 'service': 'tagging'},
    {'name': 'Bildung und Wissenschaft', 'service': 'tagging'},
    {'name': 'Kulturelles Erbe', 'service': 'tagging'},
    {'name': 'soziale Medien ', 'service': 'tagging'},
    {'name': 'Nutzung fremnder Inhalte', 'service': 'tagging'},
]


def get_dataset_1():
    return {'id': 10,
            'name': 'tagging_data_set',
            'service': 'tagging',
            'data': get_dataelements()}


def get_dataset_2():
    return {'id': 33,
            'name': 'tagging_data_set_1',
            'service': 'tagging',
            'data': get_dataelements()}


def get_dataset_3():
    return {'id': 4,
            'name': 'tagging_data_set_2',
            'service': 'tagging',
            'data': get_dataelements()}


def get_tags():
    return tags


def get_dataelements():
    return [
        {'text': 'Lorem ipsum Dolor sit amet', 'tags': ['Rechteinhaberschaft']},
        {'text': 'Tolle Idee!', 'tags': ['Bildung und Wissenschaft', 'soziale Medien ']},
        {'text': 'Idiotisch, wir sollten damit sofort aufhören.', 'tags': ['Nutzung fremnder Inhalte']}
    ]


def get_new_dataelement():
    return {'text': 'New Dataelement', 'tags': ['Respekt und Anerkennung']}
