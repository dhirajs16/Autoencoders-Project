from django.urls import path
from . import views

urlpatterns = [
    path('', views.detect_tb, name='detect_tb'),
]
