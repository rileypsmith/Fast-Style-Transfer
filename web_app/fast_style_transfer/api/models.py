from datetime import timedelta
from django.db import models
from django.utils import timezone

# Create your models here.
class Image(models.Model):
    """Basic image class for temporarily storing uploaded images"""
    image = models.ImageField()
    expiration_date = models.DateTimeField(blank=True, null=True)

    def save(self, *args, **kwargs):
        # Set expiration date to one day in the future
        self.expiration_date = timezone.now() + timedelta(days=1)
        super().save(*args, **kwargs)

class Weight(models.Model):
    """Model for connecting the proper weights file to the neural network"""
    weight_file = models.CharField(max_length=256)
