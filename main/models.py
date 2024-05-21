from django.db import models

# Create your models here.

class DetectorModel(models.Model):
    imagePath = models.ImageField(upload_to="uploads")
