# Generated by Django 4.2 on 2024-05-15 10:21

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='DetectorModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('imagePath', models.ImageField(upload_to='uploads')),
            ],
        ),
    ]