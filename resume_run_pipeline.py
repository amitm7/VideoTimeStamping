import os
import json
from scripts.extract_audio import extract_audio
from scripts.detect_scenes import get_scene_timestamps
from scripts.extract_scene_frames import extract_frames
from scripts.transcribe_whisper import transcribe
from scripts.run_ocr import run_ocr_on_images
from scripts.segment_with_llm import format_prompt, label_segments, chunk_transcript, label_segments_with_openai

video_path = "input/input_video.mp4"
audio_path = "audio/audio.wav"
scenes_dir = "scenes"
transcript_path = "transcripts/transcript.json"
ocr_path = "scenes/ocr_results.json"
output_path = "output/chapters.txt"

# 1. Extract audio
if not os.path.exists(audio_path):
    print("🔊 Extracting audio...")
    extract_audio(video_path, audio_path)
else:
    print("✅ Audio already extracted.")

# 2. Detect scenes
scenes_file = os.path.join(scenes_dir, "scene_000.jpg")
if not os.path.exists(scenes_file):
    print("🎬 Detecting scenes...")
    scenes = get_scene_timestamps(video_path)
else:
    print("✅ Scenes already detected.")
    scenes = None

# 3. Extract scene frames
if not os.path.exists(scenes_file):
    if scenes is None:
        scenes = get_scene_timestamps(video_path)
    print("🖼 Extracting scene frames...")
    extract_frames(video_path, scenes, scenes_dir)
else:
    print("✅ Scene frames already extracted.")

# 4. Transcribe audio and save transcript
if not os.path.exists(transcript_path):
    print("🧠 Running transcription...")
    transcript = transcribe(audio_path)
    os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)
    print(f"✅ Transcript saved to {transcript_path}")
else:
    print("✅ Transcript already exists.")
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

# 5. Run OCR and save results
if not os.path.exists(ocr_path):
    print("🔤 Running OCR on scene frames...")
    ocr_results = run_ocr_on_images(scenes_dir)
    os.makedirs(os.path.dirname(ocr_path), exist_ok=True)
    with open(ocr_path, "w", encoding="utf-8") as f:
        json.dump(ocr_results, f, ensure_ascii=False, indent=2)
    print(f"✅ OCR results saved to {ocr_path}")
else:
    print("✅ OCR results already exist.")
    with open(ocr_path, "r", encoding="utf-8") as f:
        ocr_results = json.load(f)

# Reduce data to 10% for testing
print("⚠️  Reducing data to 10% for testing.")
transcript_len = int(len(transcript) * 0.1)
transcript = transcript[:transcript_len]

ocr_keys = sorted(list(ocr_results.keys()))
ocr_len = int(len(ocr_keys) * 0.1)
ocr_keys_subset = ocr_keys[:ocr_len]
ocr_results = {k: ocr_results[k] for k in ocr_keys_subset}

# 6. Generate segment labels and save chapters
if not os.path.exists(output_path):
    print("🧩 Generating segment labels...")
    use_openai = os.getenv("OPENAI_API_KEY")
    if use_openai:
        # Chunk transcript for OpenAI as well
        chunk_size = 10
        segments = []
        for i, chunk in enumerate(chunk_transcript(transcript, chunk_size)):
            print(f"[INFO] Processing OpenAI chunk {i+1}")
            prompt = format_prompt(chunk, ocr_results)
            try:
                segs = label_segments_with_openai(prompt)
                if segs:
                    segments.extend(segs)
            except Exception as e:
                print(f"[ERROR] OpenAI chunk {i+1} failed: {e}")
    else:
        segments = label_segments(transcript, ocr_results, chunk_size=10, num_threads=1)
    print("📄 Saving chapters...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"{seg['start']} - {seg['end']}: {seg['label']}\n")
    print("✅ Done! Check output/chapters.txt")
else:
    print("✅ Chapters already saved.") 