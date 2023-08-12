from django import forms
from django.forms import ModelForm
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from demian.models import TextFileUpload


class UserForm(UserCreationForm):
    email = forms.EmailField(label="이메일")

    class Meta:
        model = User
        fields = ("username", "email")


class TextFileUploadForm(ModelForm):
    class Meta:
        model = TextFileUpload
        fields = ("text_file",)