# Migration Report: OpenAI + ElevenLabs → Ollama + Kokoro TTS

**Date:** 2026-09-25

## Summary

The project no longer needs `OPENAI_API_KEY` or `ELEVENLABS_API_KEY`. It now runs entirely on free services:

| Purpose | Before (paid) | After (free) |
| --- | --- | --- |
| Vision LLM (page detection, summaries, panel picking) | OpenAI `gpt-4o` | Ollama: `mistral-large-3:675b` on Ollama Cloud's free tier, or `qwen3-vl:4b-instruct` locally |
| JSON-repair model | OpenAI `gpt-3.5-turbo-0125` | Same Ollama model as above (override with `OLLAMA_TEXT_MODEL`) |
| Narration (TTS) | ElevenLabs API (voice "Adam") | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), running locally (voice `am_michael`) |
| API keys needed | `OPENAI_API_KEY`, `ELEVENLABS_API_KEY` | Optional `OLLAMA_API_KEY` (free). None for the local option |

## How the LLM works now

Ollama serves an OpenAI-compatible API, so the `openai` Python SDK is kept as the HTTP client and simply pointed at Ollama. This kept the prompts, the image message format, the retry logic (`RateLimitError`/`APIError`) and `response.usage` token counting unchanged.

The endpoint is chosen automatically:

- **`OLLAMA_API_KEY` set** → Ollama Cloud at `https://ollama.com/v1`, model `mistral-large-3:675b`.
- **No key** → local Ollama at `http://localhost:11434/v1`, model `qwen3-vl:4b-instruct` (fits a 6 GB GPU).
- `OLLAMA_BASE_URL`, `OLLAMA_VISION_MODEL` and `OLLAMA_TEXT_MODEL` override these.

Why `mistral-large-3:675b` is the cloud default: I checked the live Ollama Cloud catalogue. The models that accept images are `mistral-large-3:675b`, `kimi-k3`, `kimi-k2.6`, `gemma4:31b` and `minimax-m3`. Mistral Large 3 is the strongest of these that is **not** a "thinking" model. Thinking models spend tokens on hidden reasoning, which can use up the pipeline's `max_tokens=4096` limit before the JSON answer is written. You can switch to `kimi-k3` with `OLLAMA_VISION_MODEL=kimi-k3`.

## File-by-file changes

### `vision_analysis.py`
- Added `create_client()`, which builds an `OpenAI(base_url=..., api_key=...)` client for Ollama Cloud or local Ollama.
- Added `vision_model()` and `text_model()`, which read the model names from the environment when called. This matters because `.env` is loaded after the module is imported.
- Replaced the hard-coded `"gpt-4o"` / `"gpt-3.5-turbo-0125"` with those helpers.
- Renamed `analyze_images_with_gpt4_vision` → `analyze_images_with_vision`.
- Removed the OpenAI price constants `VISION_PRICE_PER_TOKEN` and `GPT_3_5_TURBO_PRICE_PER_TOKEN`.
- Added `parse_json()`, which strips the ```` ```json ```` fences that open models often add, so fewer responses need the JSON-repair call.
- `detect_important_pages` / `get_important_panels` now return `[]` instead of raising `KeyError` when a model returns JSON without the expected key. The pipeline already treats an empty list as "use the full pages".
- Removed OpenAI/GPT wording from the log messages.

### `movie_director.py`
- Removed the ElevenLabs `client.text_to_speech.convert(...)` call.
- Added `KokoroNarrator`, which loads `kokoro.KPipeline` once and turns each text segment into a 24 kHz WAV in memory. Settings: `KOKORO_VOICE` (default `am_michael`), `KOKORO_SPEED` (default `1.0`) and `KOKORO_LANG_CODE` (by default taken from the first letter of the voice name).
- Added `use_system_espeak()`. The `espeakng-loader` 0.2.4 wheel that Kokoro installs ships **without** the `phontab` data file, and espeak-ng then kills the Python process. When a system `espeak-ng` is installed, Kokoro now uses it instead.
- `add_narrations_to_script` now narrates segments one at a time, in a worker thread (`asyncio.to_thread`). Kokoro's pipeline isn't thread-safe, and this keeps the event loop free.
- Temporary narration files are now `.wav` instead of `.mp3`.
- The `make_movie` parameter was renamed `narration_client` → `narrator`.

### `app.py`
- Removed the `openai.OpenAI` and `elevenlabs.AsyncElevenLabs` imports and clients. It now uses `create_client()` and `KokoroNarrator()`. In `--text-only` mode Kokoro isn't loaded.
- Logs which Ollama endpoint and model are in use at startup.
- Replaced the dollar-cost printout (OpenAI and ElevenLabs prices) with plain token and character counts.
- **Bug fix (was already there before this change):** the panel-selection loop used the variable name `tokens` for each segment's count, which overwrote the summarization token total. It's now `segment_tokens`.

### `anime_recap.ipynb`
- Applied the same changes as in `app.py`: imports, client and narrator setup, function rename, token-count output and the `tokens` bug fix.
- Updated the markdown explanations to say "vision model", "Ollama" and "Kokoro" instead of GPT-Vision, GPT-3.5 and ElevenLabs.

### `requirements.txt`
- Removed `elevenlabs`.
- Added `kokoro>=0.9.4` and `soundfile`.
- Kept `openai`, which is still used as the client for Ollama's OpenAI-compatible API.

### `.env.example`
- Replaced `OPENAI_API_KEY` / `ELEVENLABS_API_KEY` with `OLLAMA_API_KEY`, plus commented-out optional overrides (Ollama URL and models, Kokoro voice and speed).

### `README.md`
- Updated the title, overview and features for Ollama and Kokoro.
- New prerequisites: **Python 3.10–3.12** (Kokoro doesn't support 3.13+ yet) and the `espeak-ng` system package.
- New setup steps for both free options (Ollama Cloud key, or local `ollama pull` + `OLLAMA_CONTEXT_LENGTH=32768 ollama serve`), plus a table of the optional settings.

## Setup (quick reference)

```bash
sudo dnf install espeak-ng            # or: sudo apt install espeak-ng
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                  # then add OLLAMA_API_KEY, or remove that line to run locally
python app.py --manga naruto --volume-number 10
```

On first run Kokoro downloads its weights (~330 MB) from Hugging Face, and misaki downloads the spaCy `en_core_web_sm` model through pip.

## Testing done

Tests ran in a throwaway Python 3.12 environment, with CPU-only torch and `kokoro` 0.9.4:

- **Kokoro → video:** two narrated segments produced a 10.1 s `recap.mp4` with an audio track (via `KokoroNarrator` → `add_narrations_to_script` → `create_movie_from_script`). The temporary WAV files were deleted afterwards.
- **Ollama (local):** ran against a local Ollama 0.34.4 server with the small `moondream` model. `create_client()` defaulted to `http://localhost:11434/v1`. Requests with base64 images went through, `usage.total_tokens` came back, the JSON-repair fallback with `response_format={"type": "json_object"}` ran, and a malformed answer came back as `[]` instead of crashing.
- **Ollama Cloud:** confirmed that `https://ollama.com/v1/chat/completions` exists and asks for a key (HTTP 401 without one), and that `mistral-large-3:675b` accepts images. **No request with a real key was made**, because no key was available.
- **Not run:** the full `app.py` pipeline on a whole volume. It needs a real vision model (the cloud key, or a pulled `qwen3-vl`), and a full volume sends hundreds of image requests.

## Known limitations / notes

- **Quality:** local 4B–8B vision models are much weaker than GPT-4o at reading manga and following the JSON format. Use the Ollama Cloud option for the best results.
- **Free-tier limits:** Ollama Cloud's free tier has usage limits. A full volume makes many requests, so you may hit rate limits. The existing retry logic will back off and retry.
- **Local context size:** each request sends about 20 page images. Start the local server with a larger context window (`OLLAMA_CONTEXT_LENGTH=32768`), or images may be cut off.
- **Python version:** your system Python is 3.14. Use a 3.10–3.12 interpreter for the venv, because pip can only find `kokoro` 0.7.x on 3.14.

---

# Part 2: Rate-limit and connection resilience (Ollama Cloud free tier)

**Date:** 2026-09-25

## Root cause

The crash (`APIConnectionError` ← `RemoteProtocolError: Server disconnected without sending a response`) came from the **important-panel selection** stage:

1. **Request burst.** `app.py` sent one request per segment through `ThreadPoolExecutor()` with no `max_workers`. That defaults to `min(32, CPUs + 4)`, which is **20 threads** on your 16-thread machine. So about 20 large, image-heavy requests hit Ollama Cloud at the same moment, and the free tier answers that with 429s and then dropped connections.
2. **Too many requests.** There were 82 panel requests, and each one also re-sent all 5 `profile-reference.pdf` pages. Any response that wasn't clean JSON cost a second "JSON repair" request.
3. **Retries made it worse.** The tenacity decorator allowed 5 attempts with waits of only 4–10 s. The OpenAI SDK also retried 2 more times on its own, so a single call could become up to 15 attempts. Each of the 20 threads retried independently, which kept the burst going instead of backing off.
4. **One failure killed the whole run.** When a worker ran out of retries, `future.result()` re-raised in the main thread. Nothing had been saved, so the finished page-detection and summary work was lost too.

## API call sites

`chat.completions.create` is called in exactly 4 places, all in `vision_analysis.py`. All of them now go through the `chat()` helper:

| Function | Stage | Called from |
| --- | --- | --- |
| `detect_important_pages` | profile/chapter page detection | `app.py`, one call per 20 pages |
| `analyze_images_with_vision` | summary / script writing | `app.py`, one call per job (7 jobs) |
| `get_important_panels_batch` (new) | important-panel selection | `panel_selection.py`, one call per batch |
| `completions` | JSON repair fallback | inside the two JSON stages above |

The old per-segment `get_important_panels` is kept for compatibility, but nothing calls it any more.

## Request count per stage (177-page volume, ~82 segments)

| Stage | Before | After |
| --- | --- | --- |
| Page detection | 9 (+ up to 9 JSON repairs), one at a time | 9, one at a time, checkpointed |
| Summaries | 7, one at a time | 7, one at a time, checkpointed |
| Panel selection | **82** (+ up to 82 JSON repairs), **~20 at once** | **~11** (measured: 11 for 84 segments), **1 at a time**, checkpointed |
| Panel selection with `--text-only` | 82 | **0** (the text file never uses panels) |
| **Total** | **~98 base, up to ~189**, times up to 15 attempts per call | **~27** |
| Reference images re-sent in panel stage | 82 × 5 = 410 | ~11 × 5 = 55 |

That's roughly a **72% cut in total requests** and an **~87% cut in panel-selection requests**. Peak concurrency drops from 20 to 1.

JSON repair calls are now rare. Both JSON stages request `response_format={"type": "json_object"}`, and `parse_json` strips the ```` ```json ```` fences that gemma4 still adds (checked against Ollama Cloud).

## Bottlenecks and recommended architecture

- **Bottleneck:** free-tier concurrency, not tokens. A single gemma4:31b request with a few images finishes in about 1–4 s. The failures came from parallel bursts.
- **Architecture now used:**
  1. **One shared throttle.** Every model call in the process goes through `api_resilience.THROTTLE` (default: 1 request at a time, at least 2 s between request starts). Even if some code uses threads, as the notebook's page-detection cell does, requests are still queued one at a time.
  2. **Batching.** Neighbouring segments are grouped into one request, capped at 40 unique panels / 8 segments. Panels shared between segments are sent once, and each panel image is labelled with its number.
  3. **Checkpoint after every request.** A crash costs at most the one request that was in flight.
  4. **Panel selection can degrade.** If it still fails after all retries, that batch falls back to full pages (the video still renders), and the next run retries only those segments.

## Retry and resume changes

- **`retry_api_call` rewritten** (moved from `app.py` to `api_resilience.py`):
  - **Retried:** `APIConnectionError` (which covers `RemoteProtocolError`, timeouts and connection resets), 408/409/425/429 and 5xx responses.
  - **Fails immediately:** 400, 401 and 404 (bad request, bad key, unknown model), instead of retrying pointlessly.
  - **Backoff:** exponential with "equal jitter", `delay = min(300, 5·2^attempt)`, sleeping between half and all of it. It honours `Retry-After` when the server sends one.
  - **Attempts:** up to 10 retries (about 20+ minutes of waiting) before giving up.
  - **Shared pause:** a failure pauses *all* threads via the throttle, not just the one that failed.
- **SDK hidden retries turned off** (`max_retries=0`), so attempts no longer multiply. The timeout is configurable (`OLLAMA_TIMEOUT_SECONDS`).
- **Checkpoint file** at `<manga>/v<volume>/checkpoint.json`:
  - **Contents:** saves page-detection results per batch, each summary as soon as it's generated, and each segment's panel picks after every panel batch.
  - **Safe writes:** writes go to a temp file that is then renamed, so a crash can't corrupt it.
  - **Stale checkpoints:** it's ignored automatically if the PDF, page count, batch size or number of jobs changes.
  - **Segment check:** each segment is matched by a hash of its text, so saved picks are never applied to the wrong segment.
  - **`--fresh`:** starts over.
- **Crash message:** if an essential stage (page detection or summaries) is still failing after all retries, the run stops with a message saying progress is saved and that re-running the same command resumes.

## Files modified in this part

| File | Change |
| --- | --- |
| `api_resilience.py` (new) | `RequestThrottle`, `retry_api_call`, `is_transient`, `backoff_delay` |
| `checkpoint.py` (new) | `Checkpoint`: the resumable JSON progress file |
| `panel_selection.py` (new) | Batch planning, panel de-duplication, checkpointed `select_important_panels` |
| `vision_analysis.py` | `chat()` helper used by all 4 call sites; SDK retries off; JSON mode; new `get_important_panels_batch`; removed the OpenAI-specific `rate_limit_exceeded` swallow |
| `prompts.py` | `KEY_PANEL_BATCH_IDENTIFICATION_INSTRUCTIONS` |
| `app.py` | Removed both thread pools and the tenacity decorator; page detection and summaries checkpointed; batched panel selection; panel stage skipped with `--text-only`; `--fresh` flag; resume hint on crash |
| `anime_recap.ipynb` | Panel cell uses `select_important_panels` (no more 20-thread pool) |
| `requirements.txt` | Removed `tenacity` (no longer used) |
| `.gitignore` | Ignore `checkpoint.json` |
| `README.md` | New settings and a "Resuming after a crash" section |

New optional settings (all have safe defaults): `OLLAMA_MAX_CONCURRENCY`, `OLLAMA_MIN_REQUEST_INTERVAL`, `OLLAMA_MAX_RETRIES`, `OLLAMA_RETRY_BASE_SECONDS`, `OLLAMA_RETRY_MAX_SECONDS`, `OLLAMA_TIMEOUT_SECONDS`, `PANEL_BATCH_MAX_IMAGES`, `PANEL_BATCH_MAX_SEGMENTS`.

## Testing done (part 2)

- **Retry unit tests:**
  - A disconnect, then a 429, then a 503 all recover.
  - A 400 fails after a single attempt.
  - Once retries run out, the last error is raised.
  - `Retry-After` is honoured.
  - The jitter stays within its bounds.
  - Five threads sharing the throttle never overlap and start at least `min_interval` apart.
- **Real Ollama Cloud (`gemma4:31b`):** one batched request (6 panels, 2 segments) returned valid picks for both segments, each from that segment's own candidates, in 3.9 s using 2,088 tokens.
- **End-to-end crash/resume** on the real 177-page `v10.pdf`, with real panel extraction and a fake API client (run in a scratch copy, not your `naruto/v10`):
  - **Run A:** the API goes down during the 4th summary. The run exits after saving 9 page batches and 3 summaries.
  - **Run B:** re-run. Pages and summaries 0–2 come from the checkpoint; only summaries 3–6 are requested. The API then goes down during panel batch 3. The 9 failed batches fall back to full pages and the run *finishes* (movie step reached).
  - **Run C:** re-run. **Only the 9 failed panel batches** are sent. No page-detection or summary requests are repeated.
- **Not tested:** a full real run on Ollama Cloud. Long outages and real rate-limit behaviour were only simulated.
