from django.urls import path

from . import views

urlpatterns = [
    path("",views.index,name="index"),
    path("detect",views.DetectView.as_view(),name="detect"),
    path("send",views.send_file)
]
