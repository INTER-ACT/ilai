from django.db import models

services = ['sentiment', 'tagging']
tags = ['PRO', 'CONTRA']

SERVICE_CHOICES = [(item, item) for item in services]
TAG_CHOICES = [(item, item) for item in tags]               # TODO: include Taglist choices


class DataSet(models.Model):
    name = models.CharField(max_length=100, default='data_set')
    service = models.CharField(choices=SERVICE_CHOICES, max_length=20)


class Tag(models.Model):
    name = models.CharField(choices=TAG_CHOICES, max_length=40, unique=True)
    service = models.CharField(choices=SERVICE_CHOICES, max_length=20)


class DataElement(models.Model):
    text = models.TextField()
    tags = models.ManyToManyField(Tag, related_name='tags')
    data_set = models.ForeignKey(DataSet, on_delete=models.CASCADE, null=True, related_name='data')
