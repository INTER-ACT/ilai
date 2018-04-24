from api.models import DataSet, DataElement, Tag, TAG_CHOICES, SERVICE_CHOICES
from rest_framework import serializers


def _dataset_ids_exist(dataset_ids):
    for id in dataset_ids:
        if not DataSet.objects.filter(pk=id).exists():
            return False
    return True


def _dataset_ids_valid_for_service(dataset_ids, service):
    for id in dataset_ids:
        if DataSet.objects.filter(pk=id).exists():
            ds = DataSet.objects.get(pk=id)
            if ds.service != service:
                return False
    return True


class _StringListField(serializers.ListField):
    child = serializers.CharField()


class _IntegerListField(serializers.ListField):
    child = serializers.IntegerField()


class _ServiceChoiceField(serializers.ChoiceField):
    def __init__(self):
        super().__init__(choices=SERVICE_CHOICES)


class _TagChoiceField(serializers.ChoiceField):
    def __init__(self):
        super().__init__(choices=TAG_CHOICES)


class _TextDataSerializer(serializers.Serializer):
    text = serializers.CharField()
    text_id = serializers.IntegerField()


""" ------- Service Serializer ------- """


class PredictSerializer(serializers.Serializer):
    texts = serializers.ListField(child=_TextDataSerializer())
    threshold = serializers.IntegerField(min_value=0, max_value=100)

    def create(self, validated_data):
        pass

    def update(self, instance, validated_data):
        pass


class TrainSerializer(serializers.Serializer):
    dataset_ids = _IntegerListField()

    def create(self, validated_data):
        pass

    def update(self, instance, validated_data):
        pass

    def dataset_ids_exist(self):
        return _dataset_ids_exist(self.data['dataset_ids'])

    def dataset_ids_valid_for_service(self, service):
        return _dataset_ids_valid_for_service(self.data['dataset_ids'], service)


class TestSerializer(serializers.Serializer):
    dataset_ids = _IntegerListField()

    def create(self, validated_data):
        pass

    def update(self, instance, validated_data):
        pass

    def dataset_ids_exist(self):
        return _dataset_ids_exist(self.data['dataset_ids'])

    def dataset_ids_valid_for_service(self, service):
        return _dataset_ids_valid_for_service(self.data['dataset_ids'], service)


class LoadSerializer(serializers.Serializer):
    model_version = serializers.ChoiceField(choices=['latest', 'before_training', 'initial'])

    def create(self, validated_data):
        pass

    def update(self, instance, validated_data):
        pass


""" ------- Data Serializer ------- """


class TagSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tag
        fields = 'name'


class DataElementSerializer(serializers.ModelSerializer):
    tags = serializers.SlugRelatedField(
        many=True,
        slug_field='name',
        queryset=Tag.objects.all()
    )

    class Meta:
        model = DataElement
        fields = ('text', 'tags')

    def create_for_dataset(self, data_set, validated_data):
        tags = validated_data.pop('tags')
        de = DataElement.objects.create(data_set=data_set, **validated_data)
        for tag in tags:
            de.tags.add(tag)
        de.save()
        validated_data['tags'] = tags


class DataSetSerializer(serializers.ModelSerializer):
    data = DataElementSerializer(many=True)

    class Meta:
        model = DataSet
        fields = ('id', 'name', 'service', 'data')

    def create(self, validated_data):
        data_elements = validated_data.pop('data')
        data_set = DataSet.objects.create(**validated_data)
        for data_element in data_elements:
            tags = data_element.pop('tags')
            de = DataElement.objects.create(data_set=data_set, **data_element)
            for tag in tags:
                de.tags.add(tag)
            de.save()
        return data_set

    def update(self, instance, validated_data):
        data_elements_new = validated_data.pop('data')
        instance.data.all().delete()

        instance.name = validated_data.get('name', instance.name)
        instance.service = validated_data.get('service', instance.service)
        instance.save()

        for data_element in data_elements_new:
            tags = data_element.pop('tags')
            de = DataElement.objects.create(data_set=instance, **data_element)
            for tag in tags:
                de.tags.add(tag)
            de.save()
        return instance


class DataSetShortSerializer(serializers.ModelSerializer):

    class Meta:
        model = DataSet
        fields = ('id', 'name', 'service')
