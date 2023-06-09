# Generated by Django 4.1 on 2023-04-27 18:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('playground', '0007_alter_doctor_doctor_id_alter_patient_patient_id'),
    ]

    operations = [
        migrations.RenameField(
            model_name='patient',
            old_name='doctor_id',
            new_name='doctor',
        ),
        migrations.RemoveField(
            model_name='patient',
            name='id',
        ),
        migrations.AlterField(
            model_name='doctor',
            name='doctor_id',
            field=models.IntegerField(default=71681, editable=False, primary_key=True, serialize=False, unique=True),
        ),
        migrations.AlterField(
            model_name='doctor',
            name='email_id',
            field=models.EmailField(help_text='We take mail-id for username', max_length=50, unique=True),
        ),
        migrations.AlterField(
            model_name='patient',
            name='patient_id',
            field=models.IntegerField(default=34539, editable=False, primary_key=True, serialize=False, unique=True),
        ),
    ]
