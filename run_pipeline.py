from scripts.extract_audio import extract_audio
from scripts.detect_scenes import get_scene_timestamps
from scripts.extract_scene_frames import extract_frames
from scripts.run_ocr import run_ocr_on_images
from scripts.transcribe_whisper import transcribe
from scripts.segment_with_llm import format_prompt, label_segments

video_path = "input/input_video.mp4"
audio_path = "audio/audio.wav"
scenes_dir = "scenes"
transcript_path = "transcripts/transcript.json"
output_path = "output/chapters.txt"

print("🔊 Extracting audio...")
extract_audio(video_path, audio_path)

print("🎬 Detecting scenes...")
scenes = get_scene_timestamps(video_path)

print("🖼 Extracting scene frames...")
extract_frames(video_path, scenes, scenes_dir)

print("🧠 Running transcription...")
transcript = transcribe(audio_path)

print("🔤 Running OCR on scene frames...")
ocr_results = run_ocr_on_images(scenes_dir)

print("🧩 Generating segment labels...")
prompt = format_prompt(transcript, ocr_results)
segments = label_segments(prompt)

print("📄 Saving chapters...")
with open(output_path, "w", encoding="utf-8") as f:
    for seg in segments:
        f.write(f"{seg['start']} - {seg['end']}: {seg['label']}\n")

print("✅ Done! Check output/chapters.txt")
