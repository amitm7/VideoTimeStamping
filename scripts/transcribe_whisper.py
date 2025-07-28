from faster_whisper import WhisperModel

def transcribe(audio_path, lang="hi"):
    model = WhisperModel("base", device="gpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, language=lang)
    transcript = []
    for seg in segments:
        transcript.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text
        })
    return transcript
