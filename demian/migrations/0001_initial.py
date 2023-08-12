# Generated by Django 4.2.4 on 2023-08-09 15:30

from django.db import migrations, models
import mysite.utils


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="TextFileUpload",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("title", models.TextField(max_length=40, null=True)),
                (
                    "text_file",
                    models.FileField(
                        blank=True,
                        null=True,
                        upload_to=mysite.utils.rename_file_to_uuid,
                    ),
                ),
                ("content", models.TextField()),
            ],
        ),
    ]
