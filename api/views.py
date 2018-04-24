from rest_framework.permissions import IsAuthenticated

from api.ai_model_controller import AIModelController
from api.apps import taggingModelController, sentimentModelController
from api.authentication import StaticTokenAuthentication
from api.models import DataSet
from api.serializers import PredictSerializer, TrainSerializer, TestSerializer, LoadSerializer, DataSetSerializer, \
    DataSetShortSerializer, DataElementSerializer
from django.http import Http404
from rest_framework import status
from rest_framework import viewsets
from rest_framework.decorators import list_route
from rest_framework.response import Response
from rest_framework.views import APIView

from exceptions.model_exceptions import LoadModelException


class ModelViewSet(viewsets.ViewSet):
    authentication_classes = (StaticTokenAuthentication,)
    permission_classes = (IsAuthenticated,)
    model_controller = None

    @list_route(methods=['post'])
    def predict(self, request, *args, **kwargs):
        serializer = PredictSerializer(data=request.data)
        if serializer.is_valid():
            return_data = self.model_controller.predict(serializer.data)
            return Response(return_data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @list_route(methods=['post'])
    def train(self, request, *args, **kwargs):
        serializer = TrainSerializer(data=request.data)
        if serializer.is_valid():
            if not serializer.dataset_ids_exist():
                return Response(data="A data set does not exist", status=status.HTTP_400_BAD_REQUEST)
            if not serializer.dataset_ids_valid_for_service(self.model_controller.service):
                return Response(data="A data set is not a valid {} data set".format(self.model_controller.service),
                                status=status.HTTP_400_BAD_REQUEST)
            else:
                self.model_controller.train(serializer.data)
                return Response(status=status.HTTP_202_ACCEPTED)

        return Response(serializer.errors, status.HTTP_400_BAD_REQUEST)

    @list_route(methods=['post'])
    def test(self, request, *args, **kwargs):
        serializer = TestSerializer(data=request.data)
        if serializer.is_valid():
            if not serializer.dataset_ids_exist():
                return Response(data="A data set does not exist", status=status.HTTP_400_BAD_REQUEST)
            if not serializer.dataset_ids_valid_for_service(self.model_controller.service):
                return Response(data="A data set is not a valid {} data set".format(self.model_controller.service),
                                status=status.HTTP_400_BAD_REQUEST)
            else:
                return_data = self.model_controller.test(serializer.data)
                return Response(data=return_data, status=status.HTTP_200_OK)

        return Response(serializer.errors, status.HTTP_400_BAD_REQUEST)

    @list_route(methods=['post'])
    def load(self, request, *args, **kwargs):
        serializer = LoadSerializer(data=request.data)
        if serializer.is_valid():
            try:
                self.model_controller.load(serializer.data)
                return Response(status=status.HTTP_202_ACCEPTED)
            except LoadModelException as e:
                return Response(data="Loading model failed due to internal errors: " + str(e),
                                status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(serializer.errors, status.HTTP_400_BAD_REQUEST)


class TaggingViewSet(ModelViewSet):
    model_controller = taggingModelController


class SentimentViewSet(ModelViewSet):
    model_controller = sentimentModelController


class DataSetList(APIView):
    authentication_classes = (StaticTokenAuthentication,)
    permission_classes = (IsAuthenticated,)

    def get(self, request, *args, **kwargs):
        data_sets = DataSet.objects.all()
        serializer = DataSetShortSerializer(data_sets, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request, *args, **kwargs):
        serializer = DataSetSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class DataSetDetail(APIView):
    authentication_classes = (StaticTokenAuthentication,)
    permission_classes = (IsAuthenticated,)

    def get_object(self, pk):
        try:
            return DataSet.objects.get(pk=pk)
        except DataSet.DoesNotExist:
            raise Http404

    def get(self, request, pk, *args, **kwargs):
        data_set = self.get_object(pk)
        serializer = DataSetSerializer(data_set)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request, pk, *args, **kwargs):
        data_set = self.get_object(pk)
        serializer = DataElementSerializer(data=request.data)
        if serializer.is_valid():
            serializer.create_for_dataset(data_set=data_set, validated_data=serializer.validated_data)
            ds_serializer = DataSetSerializer(instance=DataSet.objects.get(pk=pk))
            return Response(ds_serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def put(self, request, pk, *args, **kwargs):
        data_set = self.get_object(pk)
        serializer = DataSetSerializer(data_set, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk, *args, **kwargs):
        data_set = self.get_object(pk)
        data_set.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
