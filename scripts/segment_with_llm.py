from openai import OpenAI
import subprocess
import json
import os
import threading

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def format_prompt(transcript, ocr):
    t_text = "\n".join([f"[{t['start']:.2f} - {t['end']:.2f}]: {t['text']}" for t in transcript])
    ocr_text = "\n".join([f"{k}: {', '.join(v)}" for k, v in ocr.items()])
    prompt = f"""Below is a Hindi transcript and on-screen OCR text from a news video.

Label the segments as one of the following:
- Introduction
- Guest Introduction
- Topic A
- Topic B
- Promo
- Ad
- Summary

Transcript:
{t_text}

OCR from scenes:
{ocr_text}

Respond in JSON format like:
[
  {{"start": "00:00", "end": "01:10", "label": "Introduction"}},
  ...
]
"""
    return prompt

def label_segments_with_openai(prompt):
    response = client.chat.completions.create(model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.2)
    
    content = response.choices[0].message.content
    print(f"[DEBUG] Raw OpenAI response:\n---\n{content}\n---")
    
    # Strip markdown code block if present
    if content.startswith("```json"):
        content = content[7:-3].strip()
    elif content.startswith("```"):
        content = content[3:-3].strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to decode JSON from OpenAI: {e}")
        return []

def label_segments_with_ollama(prompt, timeout=120):
    try:
        result = subprocess.run(
            ["ollama", "run", "llama3"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        print("[ERROR] Ollama subprocess timed out.")
        return []
    except Exception as e:
        print(f"[ERROR] Ollama subprocess failed: {e}")
        return []

def chunk_transcript(transcript, chunk_size=20):
    # Split transcript into chunks of chunk_size segments
    for i in range(0, len(transcript), chunk_size):
        yield transcript[i:i+chunk_size]

# Thread worker for labeling a chunk
class LabelThread(threading.Thread):
    def __init__(self, transcript_chunk, ocr_results, results, idx):
        super().__init__()
        self.transcript_chunk = transcript_chunk
        self.ocr_results = ocr_results
        self.results = results
        self.idx = idx
    def run(self):
        print(f"[INFO] Processing chunk {self.idx+1}")
        prompt = format_prompt(self.transcript_chunk, self.ocr_results)
        segs = label_segments_with_ollama(prompt)
        self.results[self.idx] = segs

def label_segments_chunked(transcript, ocr_results, chunk_size=10, num_threads=1):
    chunks = list(chunk_transcript(transcript, chunk_size))
    results = [None] * len(chunks)
    threads = []
    for i, chunk in enumerate(chunks):
        t = LabelThread(chunk, ocr_results, results, i)
        threads.append(t)
    # Start threads in batches
    for i in range(0, len(threads), num_threads):
        batch = threads[i:i+num_threads]
        for t in batch:
            t.start()
        for t in batch:
            t.join()
    # Flatten results
    merged = []
    for segs in results:
        if segs:
            merged.extend(segs)
    return merged

# Overwrite label_segments to use chunked/threaded approach

def label_segments(prompt_or_transcript, ocr_results=None, chunk_size=10, num_threads=1):
    # If called with just a prompt (old style), fallback to old method
    if isinstance(prompt_or_transcript, str):
        if os.getenv("OPENAI_API_KEY"):
            return label_segments_with_openai(prompt_or_transcript)
        else:
            return label_segments_with_ollama(prompt_or_transcript)
    # Otherwise, assume transcript and ocr_results are provided
    if os.getenv("OPENAI_API_KEY"):
        # Use OpenAI for the whole transcript (may hit token limits)
        prompt = format_prompt(prompt_or_transcript, ocr_results)
        return label_segments_with_openai(prompt)
    else:
        return label_segments_chunked(prompt_or_transcript, ocr_results, chunk_size, num_threads)
