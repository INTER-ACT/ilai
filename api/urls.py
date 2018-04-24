from rest_framework.routers import SimpleRouter
from rest_framework.urls import url

from api import views

router = SimpleRouter(trailing_slash=False)
router.register(r'tagging', views.TaggingViewSet, base_name='tagging')
router.register(r'sentiment', views.SentimentViewSet, base_name='sentiment')

urlpatterns = [
    url(r'^datasets/$', views.DataSetList.as_view()),
    url(r'^datasets/(?P<pk>[0-9]+)/$', views.DataSetDetail.as_view())
]

urlpatterns += router.urls
