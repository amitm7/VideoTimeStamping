import ffmpeg

def extract_audio(video_path, audio_path):
    ffmpeg.input(video_path).output(audio_path, ac=1, ar=16000).run()
