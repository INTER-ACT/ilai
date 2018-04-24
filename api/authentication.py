from django.contrib.auth.models import User
from rest_framework import authentication
from rest_framework import exceptions
from api.apps import TRUE_TOKEN


class StaticTokenAuthentication(authentication.BaseAuthentication):

    def authenticate(self, request):
        token = request.META.get('HTTP_AUTHORIZATION')

        if 'token' in token:
            i = token.index('token')
            token = token[i + 6: len(token)]

        elif 'Token' in token:
            i = token.index('Token')
            token = token[i + 6: len(token)]

        if token == TRUE_TOKEN:
            user = User.objects.get_or_create(username='inter-act')[0]
            return (user, None)

        else:
            raise exceptions.AuthenticationFailed('Authorization not valid')


