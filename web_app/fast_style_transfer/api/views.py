from pathlib import Path

from django.shortcuts import render
from django.core.files.images import ImageFile
from django.core.files.uploadedfile import UploadedFile
from django.http import FileResponse
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import renderers

from .serializers import ImageSerializer
from .models import Image, Weight

import nn

# Create your views here.
@api_view(['GET'])
def image_list(request):
    """List all images in the database"""
    # if request.method == 'GET':
    images = Image.objects.all()
    serializer = ImageSerializer(images, many=True)
    return Response(serializer.data)
    # elif request.method == 'POST':

@api_view(['POST'])
def image_create(request):
    """Create a new image instance"""
    serializer = ImageSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
def image_view(request, pk):
    """Retrieve a specific image for a user to view"""
    image = Image.objects.get(id=pk)
    serializer = ImageSerializer(image)
    return Response(serializer.data)

@api_view(['POST'])
def run(request):
    """Call the actual neural network processing"""
    # Fetch image from database
    image = Image.objects.get(id=int(request.data['image_id']))
    # Fetch the weight model for the network
    weights = Weight.objects.get(id=int(request.data['weight_id']))
    # Make config dictionary to pass to run function
    cfg = {
        'image_path': image.image.url,
        'weight_path': weights.weight_file,
        'vangoh': (int(request.data['weight_id']) == 1)
    }
    # Run it and save output
    out_path = nn.process(cfg)
    # Create new image model from that stored image
    out_image = Image.objects.create(
        image = UploadedFile(file=open(out_path, 'rb'))
    )
    # out_image.image = ImageFile(open(out_path, 'rb'))
    out_image.save()
    # Return serialized image
    serializer = ImageSerializer(out_image)
    return Response(serializer.data)

class PassthroughRenderer(renderers.BaseRenderer):
    """
        Return data as-is. View should supply a Response.
    """
    media_type = ''
    format = ''
    def render(self, data, accepted_media_type=None, renderer_context=None):
        return data

@api_view(['POST'])
def download(request):
    """Download an image (for saving image with style transfer)"""
    # Fetch the requested image from the database
    image = Image.objects.get(id=int(request.data['image_id']))

    response = FileResponse(image.image.open(), content_type='image/jpeg')
    response['Content-Length'] = image.image.size
    response['Content-Disposition'] = f'attachment; filename={Path(image.image.path).name}'

    return response
