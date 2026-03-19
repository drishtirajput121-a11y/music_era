from django import forms


class AudioUploadForm(forms.Form):
    audio_file = forms.FileField(
        label="Select your WAV/MP3 file",
        widget=forms.ClearableFileInput(attrs={"accept": ".wav,.mp3"}),
    )