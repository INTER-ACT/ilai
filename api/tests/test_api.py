from rest_framework import status
from rest_framework.test import APITestCase

from api.models import DataSet, DataElement, Tag
from api.tests import data_factory_sentiment as dfs
from api.tests import data_factory_tagging as dft
from api.apps import TRUE_TOKEN


def _createDataset(data_set):
    data_elements = data_set.pop('data')
    ds = DataSet.objects.create(**data_set)
    for data_element in data_elements:
        tags = data_element.pop('tags')
        de = DataElement.objects.create(data_set=ds, **data_element)
        for tag in tags:
            pk = Tag.objects.get(name=tag).pk
            de.tags.add(pk)
        de.save()


class TaggingServiceTests(APITestCase):
    def setUp(self):
        self.client.credentials(HTTP_AUTHORIZATION=TRUE_TOKEN)

    @classmethod
    def setUpTestData(cls):
        tags = dft.get_tags()
        for tag in tags:
            Tag.objects.create(**tag)

        _createDataset(dft.get_dataset_1())
        _createDataset(dft.get_dataset_2())
        _createDataset(dft.get_dataset_3())

    def test_predict(self):
        url = '/tagging/predict'
        data = {
            'texts': [
                {
                    'text_id': 1,
                    'text': 'Lorem ipsum Dolor sit amet'
                },
                {
                    'text_id': 2,
                    'text': 'Lorem ipsum Dolor sit amet'
                }
            ],
            'threshold': 40
        }
        response_data = [
            {
                'text_id': 1,
                'tags': []
            },
            {
                'text_id': 2,
                'tags': []
            }
        ]
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data, response_data)

    def test_train(self):
        url = '/tagging/train'
        data = {
            'dataset_ids': [10, 4, 33]
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)

    def test_test(self):
        url = '/tagging/test'
        data = {
            'dataset_ids': [10, 4, 33]
        }
        response_keys = ['accuracy', 'f1', 'precision', 'recall']

        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(list(response.data.keys()), response_keys)

    def test_load(self):
        url = '/tagging/load'
        data = {
            'model_version': 'before_training'
        }

        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)


class SentimentServiceTests(APITestCase):
    def setUp(self):
        self.client.credentials(HTTP_AUTHORIZATION=TRUE_TOKEN)

    @classmethod
    def setUpTestData(cls):
        tags = dfs.get_tags()
        for tag in tags:
            Tag.objects.create(**tag)

        _createDataset(dfs.get_dataset_1())
        _createDataset(dfs.get_dataset_2())
        _createDataset(dfs.get_dataset_3())

    def test_predict(self):
        url = '/sentiment/predict'
        data = {
            'texts': [
                {
                    'text_id': 1,
                    'text': 'Lorem ipsum Dolor sit amet'
                },
                {
                    'text_id': 2,
                    'text': 'Lorem ipsum Dolor sit amet'
                }
            ],
            'threshold': 40
        }
        response_data = [
            {
                'text_id': 1,
                'tags': ['PRO']
            },
            {
                'text_id': 2,
                'tags': ['PRO']
            }
        ]
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data, response_data)

    def test_train(self):
        url = '/sentiment/train'
        data = {
            'dataset_ids': [10, 4, 33]
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)

    def test_test(self):
        url = '/sentiment/test'
        data = {
            'dataset_ids': [10, 4, 33]
        }
        response_keys = ['accuracy', 'f1', 'precision', 'recall']

        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(list(response.data.keys()), response_keys)

    def test_load(self):
        url = '/sentiment/load'
        data = {
            'model_version': 'before_training'
        }

        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)


class DataTests(APITestCase):
    def setUp(self):
        self.client.credentials(HTTP_AUTHORIZATION=TRUE_TOKEN)

    @classmethod
    def setUpTestData(cls):
        tags = dfs.get_tags()
        for tag in tags:
            Tag.objects.create(**tag)

        _createDataset(dfs.get_dataset_1())
        _createDataset(dfs.get_dataset_2())
        _createDataset(dfs.get_dataset_3())

    def test_datasets_get(self):
        url = '/datasets/'

        response = self.client.get(url, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 3)

    def test_datasets_post(self):
        url = '/datasets/'
        data = dfs.get_dataset_1()
        data['id'] = 1

        response = self.client.post(url, data, format='json')

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(len(response.data), 4)
        self.assertEqual(DataSet.objects.get(pk=response.data['id']).name, data['name'])

    def test_dataset_detail_get(self):
        url = '/datasets/10/'
        response_data = dfs.get_dataset_1()

        response = self.client.get(url, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(dict(response.data), response_data)

    def test_dataset_detail_put(self):
        url = '/datasets/10/'
        data = dfs.get_dataset_1()
        data['name'] = 'new Name'

        response = self.client.put(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        ds = DataSet.objects.get(pk=10)
        self.assertEqual(response.data['name'], ds.name)

    def test_dataset_detail_post(self):
        url = '/datasets/10/'
        data_element = dfs.get_new_dataelement()

        response = self.client.post(url, data_element, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(len(response.data['data']), 4)
        self.assertEqual(len(DataSet.objects.get(pk=10).data.all()), 4)

    def test_dataset_detail_delete(self):
        url = '/datasets/10/'

        response = self.client.delete(url, format='json')
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
        self.assertEqual(False, DataSet.objects.filter(pk=10).exists())
        self.assertRaises(DataSet.DoesNotExist, lambda: DataSet.objects.get(pk=10))  # TODO: decide
