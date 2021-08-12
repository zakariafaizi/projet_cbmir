# forms.py
from django import forms
from .models import *

class MedicalForm(forms.ModelForm):

	class Meta:
		model = MedicalImage
		fields = ['medical_Img']

