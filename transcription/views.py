from __future__ import annotations
from .forms import AudioUploadForm

import os
import tempfile
from typing import Any

import crepe
import librosa
import numpy as np
from django.core.files.uploadedfile import UploadedFile


def predict_pitch_10ms_from_uploaded_wav(
    uploaded_wav: UploadedFile,
    *,
    sr: int = 16000,
    viterbi: bool = True,
) -> dict[str, Any]:
    """
    Load an uploaded .wav with librosa and run CREPE pitch prediction every 10ms.

    Returns CREPE outputs:
      - time: (N,) seconds
      - frequency: (N,) Hz
      - confidence: (N,) [0..1]
      - activation: (N, 360) model activation (optional downstream use)
      - sr: int (audio sample rate used for CREPE)
    """

    # CREPE expects mono float audio at 16 kHz.
    # Writing to a temp file is the most compatible approach for Django uploads on Windows.
    suffix = os.path.splitext(uploaded_wav.name or "")[1].lower()
    if suffix and suffix != ".wav":
        raise ValueError("Expected a .wav file")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        for chunk in uploaded_wav.chunks():
            tmp.write(chunk)

    try:
        audio, _sr = librosa.load(tmp_path, sr=sr, mono=True)
        if audio.size == 0:
            raise ValueError("Empty audio")

        audio = audio.astype(np.float32, copy=False)

        time, frequency, confidence, activation = crepe.predict(
            audio,
            sr,
            viterbi=viterbi,
            step_size=10,  # milliseconds
        )

        return {
            "time": time,
            "frequency": frequency,
            "confidence": confidence,
            "activation": activation,
            "sr": sr,
        }
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
def upload_audio(request):
    if request.method == 'POST':
        form = AudioUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # This calls the AI function the Cursor AI just wrote for you!
            results = predict_pitch_10ms_from_uploaded_wav(request.FILES['audio_file'])
            return render(request, 'transcription/results.html', {'results': results})
    else:
        form = AudioUploadForm()
    return render(request, 'transcription/upload.html', {'form': form})