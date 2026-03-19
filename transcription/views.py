
from __future__ import annotations
from .forms import AudioUploadForm
from django.shortcuts import render, redirect
from .utils import hz_to_note, MusicEngine
import os
import tempfile
from typing import Any
from django.core.files.uploadedfile import UploadedFile
import uuid
from django.conf import settings


def predict_pitch_10ms_from_uploaded_wav(
    uploaded_audio: UploadedFile,
    *,
    sr: int = 16000,
    viterbi: bool = True,
) -> dict[str, Any]:
    """
    Load an uploaded .wav or .mp3 with librosa and run CREPE pitch prediction every 10ms.

    Returns CREPE outputs:
      - time: (N,) seconds
      - frequency: (N,) Hz
      - confidence: (N,) [0..1]
      - activation: (N, 360) model activation (optional downstream use)
      - sr: int (audio sample rate used for CREPE)
    """

    # CREPE expects mono float audio at 16 kHz.
    # Writing to a temp file is the most compatible approach for Django uploads on Windows.
    suffix = os.path.splitext(uploaded_audio.name or "")[1].lower()
    if suffix not in {".wav", ".mp3"}:
        raise ValueError("Expected a .wav or .mp3 file")

    # Preserve original extension so librosa/ffmpeg can decode non‑WAV formats.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        for chunk in uploaded_audio.chunks():
            tmp.write(chunk)

    try:
        # Import heavy deps lazily so Django can start even if they are not installed
        # in the environment running management commands.
        import crepe  # type: ignore
        import librosa  # type: ignore
        import numpy as np  # type: ignore

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
            instrument = request.POST.get("instrument") or "Acoustic Guitar"

            # 1. Run the AI prediction
            results = predict_pitch_10ms_from_uploaded_wav(request.FILES['audio_file'])
            
            # 2. Convert frequencies to notes
            notes = [hz_to_note(f) for f in results['frequency']]
            
            # 3. Add notes and the combined list to the results dictionary
            results['notes'] = notes
            results['results_list'] = list(
                zip(
                    results['time'],
                    results['frequency'],
                    results['confidence'],
                    notes,
                )
            )

            engine = MusicEngine(instrument)
            lead_events = engine.transform(results["results_list"])

            # 5. Save the uploaded audio file so the browser can play it back
            upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            suffix = os.path.splitext(request.FILES['audio_file'].name)[1].lower()
            unique_name = uuid.uuid4().hex + suffix
            save_path = os.path.join(upload_dir, unique_name)
            request.FILES['audio_file'].seek(0)          # rewind after temp-file copy
            with open(save_path, 'wb') as f:
                for chunk in request.FILES['audio_file'].chunks():
                    f.write(chunk)
            audio_url = settings.MEDIA_URL + 'uploads/' + unique_name

            # 4. Pass the whole dictionary to the template
            return render(
                request,
                'transcription/results.html',
                {
                    'results': results,
                    'instrument': instrument,
                    'lead_events': lead_events,
                    'audio_url': audio_url,
                },
            )
    else:
        form = AudioUploadForm()
    return render(request, 'transcription/upload.html', {'form': form})