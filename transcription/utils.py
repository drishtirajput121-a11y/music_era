from music21 import pitch

def hz_to_note(hz):
    if hz <= 0:
        return "Silence"
    try:
        p = pitch.Pitch()
        p.frequency = hz
        return p.nameWithOctave  # Returns something like "C#4"
    except:
        return "Unknown"