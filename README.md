# VideoTimeStamping

This project provides a full pipeline for segmenting a news video into labeled chapters using audio transcription, scene detection, OCR, and LLM-based segment labeling.

## Features
- **Audio Extraction**: Extracts audio from the input video.
- **Scene Detection**: Detects scene changes in the video.
- **Frame Extraction**: Extracts representative frames for each scene.
- **Transcription**: Transcribes the audio using Whisper.
- **OCR**: Runs OCR on scene frames to extract on-screen text.
- **Segment Labeling**: Uses OpenAI GPT-4o or local Llama 3 (via Ollama) to label video segments (Introduction, Guest Introduction, Topic A/B, Promo, Ad, Summary, etc.).
- **Resumable Pipeline**: The `resume_run_pipeline.py` script checks for completed steps and only runs what is needed, so you can resume after interruptions.

## Requirements
- Python 3.8+
- ffmpeg (installed and in PATH)
- Ollama (for local Llama 3 support, optional)
- OpenAI API key (for GPT-4o support, optional)
- See `requirements.txt` for Python dependencies

## Setup & Usage

1. **Clone the repository and navigate to the project directory:**
   ```bash
   git clone <your-repo-url>
   cd full_m4_segmenter
   ```
2. **(Optional but recommended) Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Place your input video:**
   - Put your video file in the `input/` directory as `input_video.mp4` (or update the script paths).

### Running the Pipeline

#### Recommended: Resumable Pipeline
This script will automatically skip completed steps and resume from where it left off.

```bash
python resume_run_pipeline.py
```

- The script will use OpenAI GPT-4o if `OPENAI_API_KEY` is set, otherwise it will use local Llama 3 via Ollama.
- Output chapters will be saved to `output/chapters.txt`.

#### Original Pipeline (runs all steps from scratch)
```bash
python run_pipeline.py
```

- This will re-run all steps, even if outputs already exist.

### Example: Setting the OpenAI API Key
```bash
export OPENAI_API_KEY=sk-...yourkey...
python resume_run_pipeline.py
```

### Example: Using Local Llama 3 (Ollama)
- Make sure you have [Ollama](https://ollama.com/) installed and the Llama 3 model pulled:
  ```bash
  ollama pull llama3
  ```
- Then simply run:
  ```bash
  python resume_run_pipeline.py
  ```

## Notes
- If you hit OpenAI rate limits, the script will automatically chunk the transcript and make multiple API calls.
- For large jobs, you can use local Llama 3 via Ollama for unlimited processing (install Ollama and pull the Llama 3 model).
- You can adjust chunk size and threading in the script for performance tuning.

## File Structure
- `run_pipeline.py`: Original pipeline script (runs all steps from scratch)
- `resume_run_pipeline.py`: Resumable, robust pipeline script (recommended)
- `scripts/`: Contains all processing modules
- `input/`: Place your input video here
- `audio/`, `scenes/`, `transcripts/`, `output/`: Intermediate and final outputs

## License
MIT
