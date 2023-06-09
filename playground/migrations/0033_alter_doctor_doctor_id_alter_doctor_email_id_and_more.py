# Generated by Django 4.1 on 2023-04-29 05:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('playground', '0032_alter_doctor_doctor_id_alter_report_report_id'),
    ]

    operations = [
        migrations.AlterField(
            model_name='doctor',
            name='doctor_id',
            field=models.IntegerField(default=31358, editable=False, primary_key=True, serialize=False, unique=True),
        ),
        migrations.AlterField(
            model_name='doctor',
            name='email_id',
            field=models.EmailField(max_length=50, unique=True),
        ),
        migrations.AlterField(
            model_name='report',
            name='report_id',
            field=models.IntegerField(default=42207, editable=False, primary_key=True, serialize=False, unique=True),
        ),
    ]
