from django.urls import path #,include
from . import views
#from rest_framework import routers

#router = routers.DefaultRouter(trailing_slash=False)
#router.register("demian", views.UserViewSet)

app_name = "demian"

urlpatterns = [
    path("", views.index, name="index"),
    path('post/', views.post, name="post"),
    path('fileupload/', views.fileUpload, name="fileupload"),
]