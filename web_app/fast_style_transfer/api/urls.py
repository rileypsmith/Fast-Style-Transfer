from django.urls import path, include

from . import views

urlpatterns = [
    path('list/', views.image_list),
    path('create/', views.image_create),
    path('view/<pk>', views.image_view),
    path('run/', views.run),
    path('download/', views.download)
]
