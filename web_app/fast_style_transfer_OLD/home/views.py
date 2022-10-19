from django.shortcuts import render
from .models import Image
from .serializers import ImageSerializer
from rest_framework import generics

# Create your views here.
class ImageListCreate(generics.ListCreateAPIView):
    queryset = Image.objects.all()
    serializer_class = ImageSerializer
