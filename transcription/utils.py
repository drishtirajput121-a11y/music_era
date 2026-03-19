from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

try:
    from music21 import pitch  # type: ignore
except Exception:  # pragma: no cover
    pitch = None


def hz_to_note(hz: float) -> str:
    if hz <= 0:
        return "Silence"
    try:
        if pitch is not None:
            p = pitch.Pitch()
            p.frequency = float(hz)
            return p.nameWithOctave  # e.g. "C#4"

        # Fallback (no music21): nearest equal-temperament note name
        import math

        a4 = 440.0
        midi = int(round(69 + 12 * math.log2(float(hz) / a4)))
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        name = names[midi % 12]
        octave = (midi // 12) - 1
        return f"{name}{octave}"
    except Exception:
        return "Unknown"


_NOTE_TO_PC = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}


def _note_name_no_octave(note: str) -> str:
    # "C#4" -> "C#"; "Silence"/"Unknown" unchanged
    if not note or note[0] not in "ABCDEFG":
        return note
    # note names in music21: e.g. "C#4", "Bb3"
    i = 1
    if len(note) > 1 and note[1] in {"#", "b"}:
        i = 2
    return note[:i]


def _pc(note: str) -> int | None:
    name = _note_name_no_octave(note)
    return _NOTE_TO_PC.get(name)


def _triad_name(pcs: set[int]) -> str | None:
    # Basic major/minor triads only, returned as e.g. "C Major", "A Minor"
    # pcs contains pitch classes [0..11]
    if len(pcs) < 3:
        return None
    for root_name, root_pc in _NOTE_TO_PC.items():
        if root_pc not in pcs:
            continue
        maj = {root_pc, (root_pc + 4) % 12, (root_pc + 7) % 12}
        if maj.issubset(pcs):
            # Prefer natural root spellings for display
            disp = root_name.replace("Db", "C#").replace("Eb", "D#").replace("Gb", "F#").replace("Ab", "G#").replace("Bb", "A#")
            return f"{disp} Major"
        min_ = {root_pc, (root_pc + 3) % 12, (root_pc + 7) % 12}
        if min_.issubset(pcs):
            disp = root_name.replace("Db", "C#").replace("Eb", "D#").replace("Gb", "F#").replace("Ab", "G#").replace("Bb", "A#")
            return f"{disp} Minor"
    return None


_GUITAR_CHORD_FINGERINGS: dict[str, list[Any]] = {
    # Standard 6-string fingerings: [E A D G B e] where ints are frets, "X" is mute, 0 is open
    "C Major": ["X", 3, 2, 0, 1, 0],
    "G Major": [3, 2, 0, 0, 0, 3],
    "D Major": ["X", "X", 0, 2, 3, 2],
    "A Major": ["X", 0, 2, 2, 2, 0],
    "E Major": [0, 2, 2, 1, 0, 0],
    "A Minor": ["X", 0, 2, 2, 1, 0],
    "E Minor": [0, 2, 2, 0, 0, 0],
    "D Minor": ["X", "X", 0, 2, 3, 1],
}


@dataclass(frozen=True)
class LeadEvent:
    time: float
    note: str
    chord: str | None
    confidence: float
    frequency: float
    finger_positions: list[Any] | None
    rhythm: str | None  # e.g. "↓" or "↑"


class MusicEngine:
    """
    Lightweight logic-to-lead-sheet transformer.

    Input: timed pitch frames (time, frequency, confidence, note)
    Output: list of dict-like objects for templates/JS rendering
    """

    def __init__(
        self,
        instrument: str,
        *,
        conf_threshold: float = 0.6,
        min_note_change_semitones: int = 1,
        chord_window_s: float = 0.25,
        max_events: int = 800,
    ) -> None:
        self.instrument = instrument
        self.conf_threshold = conf_threshold
        self.min_note_change_semitones = min_note_change_semitones
        self.chord_window_s = chord_window_s
        self.max_events = max_events

    def _midi(self, note: str) -> int | None:
        if note in {"Silence", "Unknown"}:
            return None
        try:
            if pitch is not None:
                p = pitch.Pitch(note)
                return int(p.midi)

            # Fallback parse like "C#4"
            name = _note_name_no_octave(note)
            pcv = _NOTE_TO_PC.get(name)
            if pcv is None:
                return None
            # extract trailing integer octave
            import re

            m = re.search(r"(-?\d+)$", note)
            if not m:
                return None
            octave = int(m.group(1))
            return (octave + 1) * 12 + pcv
        except Exception:
            return None

    def _finger_positions(self, chord: str | None) -> list[Any] | None:
        if not chord:
            return None
        if self.instrument in {"Acoustic Guitar", "Electric Guitar"}:
            return _GUITAR_CHORD_FINGERINGS.get(chord)
        # For keys/violin we keep diagrams note-driven (JS), so no static fingering list here.
        return None

    def _rhythm_arrow(self, t: float, bpm: float = 120.0) -> str:
        # Simple alternating down/up on half-beats (demo-friendly)
        beat_s = 60.0 / bpm
        half = beat_s / 2.0
        idx = int(t / half)
        return "↓" if idx % 2 == 0 else "↑"

    def transform(self, frames: Iterable[tuple[float, float, float, str]]) -> list[dict[str, Any]]:
        events: list[LeadEvent] = []

        recent_pcs: list[tuple[float, int]] = []
        last_midi: int | None = None

        for (t, freq, conf, note) in frames:
            if len(events) >= self.max_events:
                break
            if conf is None or float(conf) < self.conf_threshold:
                continue
            if not note or note in {"Silence", "Unknown"}:
                continue

            midi = self._midi(note)
            if midi is None:
                continue
            if last_midi is not None and abs(midi - last_midi) < self.min_note_change_semitones:
                # Skip micro-jitter in pitch tracking
                continue
            last_midi = midi

            pcv = _pc(note)
            if pcv is not None:
                recent_pcs.append((float(t), pcv))
                cutoff = float(t) - self.chord_window_s
                while recent_pcs and recent_pcs[0][0] < cutoff:
                    recent_pcs.pop(0)

            chord = _triad_name({pc for _, pc in recent_pcs if pc is not None})
            if chord is None:
                # Monophonic fallback: if the current note name matches a known chord root,
                # surface that chord so the UI isn't empty (demo-friendly).
                root = _note_name_no_octave(str(note))
                candidate = f"{root} Major"
                if candidate in _GUITAR_CHORD_FINGERINGS:
                    chord = candidate
            events.append(
                LeadEvent(
                    time=float(t),
                    note=str(note),
                    chord=chord,
                    confidence=float(conf),
                    frequency=float(freq),
                    finger_positions=self._finger_positions(chord),
                    rhythm=self._rhythm_arrow(float(t)),
                )
            )

        # Return plain dicts for templates + json_script
        return [
            {
                "time": e.time,
                "note": e.note,
                "chord": e.chord,
                "confidence": e.confidence,
                "frequency": e.frequency,
                "finger_positions": e.finger_positions,
                "rhythm": e.rhythm,
                "instrument": self.instrument,
            }
            for e in events
        ]