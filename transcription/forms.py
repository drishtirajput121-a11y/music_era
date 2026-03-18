from django import forms

class AudioUploadForm(forms.Form):
    audio_file = forms.FileField(label="Select your Guitar/Piano WAV file")