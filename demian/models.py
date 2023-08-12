import os
import uuid
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from mysite.utils import rename_file_to_uuid

# Create your models here.
class TextFileUpload(models.Model):
    #owner = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.TextField(max_length=40, null=True)
    text_file = models.FileField(null=True, upload_to=rename_file_to_uuid, blank=True)
    content = models.TextField()

    def __str__(self):
        return self.text_file.name
