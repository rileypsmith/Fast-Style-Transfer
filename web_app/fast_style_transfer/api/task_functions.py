from django.utils import timezone

from django_q.models import Schedule

from .models import Image

def clean_dataset():
    Image.objects.filter(expiration_date__gte=timezone.now()).delete()
