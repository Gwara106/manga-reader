# Manga Recap with Ollama Vision + Kokoro TTS

## Project Overview

This project aims to generate summaries of manga volumes by analyzing images extracted from PDF files of the manga. It uses a vision-capable LLM served by [Ollama](https://ollama.com) (free: locally or on the Ollama Cloud free tier) to understand the content of manga pages and produce compelling, story-telling tone summaries. The project processes PDFs to extract images, scales them to a specific size, encodes them in base64, and then uses these images as input for the vision model alongside custom prompts to generate summaries. Once a summary is generated, it is narrated locally with [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M), a free open-weight text-to-speech model. The resulting narration and relevant panel images are then combined to create a video recap summarizing the volume.

The whole pipeline runs for free: no paid API keys are required.

Join the Discord: https://discord.gg/MMqcuDe2WZ

https://github.com/pashpashpash/manga-reader/assets/20898225/debb0c15-3579-477c-813d-2ed878b0e6ea

## Features

- PDF processing to extract manga pages as images as well as panel extraction from within pages.
- Image scaling to fit the requirements of the vision model.
- Base64 encoding of images for API submission.
- Generating text summaries of manga volumes in a story-telling tone.
- Narration of the generated summaries using Kokoro TTS (runs locally).
- Video creation from the narration and relevant panel images/pages.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.10–3.12 (Kokoro does not yet support Python 3.13+)
- [espeak-ng](https://github.com/espeak-ng/espeak-ng) (used by Kokoro for out-of-dictionary words), e.g. `sudo dnf install espeak-ng` / `sudo apt install espeak-ng`
- [Ollama](https://ollama.com/download) (only needed for the local option below)
- Pip3 (Python package manager)
- Virtual environment (recommended)

## Installation Steps

1. Create a virtual environment to manage your project's dependencies separately.

```
python3.12 -m venv venv
```

2. Activate the virtual environment

```
source venv/bin/activate
```

3. Install Required Python Packages

```
pip3 install -r requirements.txt
```

4. Set Up Environment Variables

Copy `.env.example` to `.env` and pick one of the two free LLM options:

**Option A – Ollama Cloud (recommended, best quality).** Create a free API key at https://ollama.com/settings/keys and add it:

```
OLLAMA_API_KEY=your_ollama_api_key_here
```

This uses `mistral-large-3:675b` (a vision model) on Ollama Cloud by default. The free tier has usage limits.

**Option B – fully local.** Leave `OLLAMA_API_KEY` unset, then pull a vision model and start Ollama with a large enough context window (each request contains ~20 images):

```
ollama pull qwen3-vl:4b-instruct
OLLAMA_CONTEXT_LENGTH=32768 ollama serve
```

`qwen3-vl:4b-instruct` fits in ~6 GB of VRAM. If you have more, `qwen3-vl:8b-instruct` gives better results.

Optional settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `https://ollama.com/v1` with a key, else `http://localhost:11434/v1` | Ollama endpoint |
| `OLLAMA_VISION_MODEL` | `mistral-large-3:675b` (cloud) / `qwen3-vl:4b-instruct` (local) | Any vision-capable Ollama model |
| `OLLAMA_TEXT_MODEL` | same as vision model | Model used to repair malformed JSON |
| `KOKORO_VOICE` | `am_michael` | Any [Kokoro voice](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md) |
| `KOKORO_SPEED` | `1.0` | Narration speed |
| `OLLAMA_MAX_CONCURRENCY` | `1` | Max simultaneous API requests (keep at 1 on the free tier) |
| `OLLAMA_MIN_REQUEST_INTERVAL` | `2` | Minimum seconds between request starts |
| `OLLAMA_MAX_RETRIES` | `10` | Retries for connection errors, rate limits and 5xx responses |
| `OLLAMA_RETRY_BASE_SECONDS` / `OLLAMA_RETRY_MAX_SECONDS` | `5` / `300` | Exponential backoff (with jitter) start and cap |
| `OLLAMA_TIMEOUT_SECONDS` | `600` | Per-request timeout |
| `PANEL_BATCH_MAX_IMAGES` / `PANEL_BATCH_MAX_SEGMENTS` | `40` / `8` | Size of each batched panel-selection request |

Kokoro needs no API key. Its model weights (~330 MB) are downloaded from Hugging Face on first run.

5. Prepare Your Manga PDFs

Place your manga volume PDF files in a directory structure as expected by the script, for example, `naruto/v10/v10.pdf`. Additionally, you should have a `chapter-reference.pdf` and a `profile-reference.pdf` in each manga directory. For example, `naruto/chapter-reference.pdf` and `naruto/profile-reference.pdf`. These files are used by the vision model to identify the chapter pages and character introductions, respectively, so that jobs can be split up by chapter and for characters to be identified correctly by the vision model.

## Running the Project

To run the project, execute the `app.py` script from the root directory of your project:

```
python3 app.py --manga naruto --volume-number 10
```

This script processes the specified PDF files, extracts and scales images, encodes them in base64, and sends them to the Ollama vision model for analysis. The summaries generated by the model are printed to the console, including the total tokens used. The script then narrates the summaries locally with Kokoro TTS. The resulting narration and relevant panel images are then combined to create a video recap summarizing the volume. The video is saved inside the relevant volume directory, i.e. `naruto/v10/recap.mp4`.

### Resuming after a crash

Every API result is saved to `<manga>/v<volume>/checkpoint.json` as soon as it arrives. If a run stops (for example Ollama Cloud is down for longer than the retries cover), re-run the same command: finished page batches, summaries and panel selections are reused and not requested again. Pass `--fresh` to ignore the checkpoint and start over.

## Optional/Recommended running instructions

I personally recommend running this in a Jupiter notebook ([anime_recap.ipynb](anime_recap.ipynb)), as it allows you run the script one cell at a time, which is useful for debugging and understanding the process.
