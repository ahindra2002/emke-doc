# Generated by Django 4.1 on 2023-04-28 07:00

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('playground', '0024_alter_doctor_doctor_id_alter_patient_patient_id_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='doctor',
            name='doctor_id',
            field=models.IntegerField(default=22297, editable=False, primary_key=True, serialize=False, unique=True),
        ),
        migrations.AlterField(
            model_name='patient',
            name='patient_id',
            field=models.IntegerField(default=70444, editable=False, primary_key=True, serialize=False, unique=True),
        ),
        migrations.CreateModel(
            name='Report',
            fields=[
                ('report_id', models.IntegerField(default=16632, editable=False, primary_key=True, serialize=False, unique=True)),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
                ('Report_text', models.CharField(max_length=300)),
                ('Report_URL', models.CharField(max_length=400)),
                ('disease_name', models.CharField(max_length=100)),
                ('doctor_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='playground.doctor')),
                ('patient_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='playground.patient')),
            ],
            options={
                'db_table': 'Report',
            },
        ),
    ]
