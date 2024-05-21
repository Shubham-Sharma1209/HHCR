from curses import meta
from . import models
from rest_framework import serializers

class DetectModelSerializer(serializers.ModelSerializer):

    class Meta:
        model = models.DetectorModel
        fields = ["imagePath"]

