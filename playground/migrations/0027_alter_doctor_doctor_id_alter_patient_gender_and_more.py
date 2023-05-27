# Generated by Django 4.1 on 2023-04-28 12:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('playground', '0026_report_image_url_alter_doctor_doctor_id_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='doctor',
            name='doctor_id',
            field=models.IntegerField(default=14949, editable=False, primary_key=True, serialize=False, unique=True),
        ),
        migrations.AlterField(
            model_name='patient',
            name='Gender',
            field=models.CharField(choices=[('Male', 'Male'), ('Female', 'Female'), ('Other', 'Other')], max_length=10),
        ),
        migrations.AlterField(
            model_name='patient',
            name='patient_id',
            field=models.IntegerField(default=16248, editable=False, primary_key=True, serialize=False, unique=True),
        ),
        migrations.AlterField(
            model_name='report',
            name='disease_name',
            field=models.CharField(choices=[('Skin Cancer', 'Skin Cancer'), ('Pneumonia', 'Pneumonia'), ('Malaria', 'Malaria'), ('Brain Cancer', 'Brain Cancer')], max_length=100),
        ),
        migrations.AlterField(
            model_name='report',
            name='report_id',
            field=models.IntegerField(default=28171, editable=False, primary_key=True, serialize=False, unique=True),
        ),
    ]
