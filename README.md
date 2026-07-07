# CleanCut

CleanCut is a console application that **automatically detects and cuts out** profanity and inappropriate content from songs.

The app downloads the song's lyrics, aligns them with the audio at the word level, uses a local AI model to detect inappropriate words, and removes those sections seamlessly using a crossfade adjusted to the song's BPM.

> ⚠️ This project is currently under development. Some files mentioned below (`requirements.txt`, `.env.example`) are **not yet included** in the repo — this README explains how to create them yourself.

---

## How It Works

1. You pick a song from the `music/` folder.
2. The song's lyrics are downloaded via the **Genius API**.
3. **WhisperX** aligns the audio file with the downloaded lyrics, producing word-level timestamps.
4. A locally running **Ollama + Gemma3** model scans the lyrics and flags profane/inappropriate phrases.
5. The exact time range of each flagged phrase in the audio is located.
6. **librosa** calculates the song's BPM, the cut points are snapped to the nearest beat, and **pydub** applies a BPM-adjusted crossfade to cut them out.
7. The result is saved to the `cleancutted/` folder.

Since the cutting is done on the original (uncompressed) audio, sound quality is preserved.

---

## Project Structure

Here's what each file in the repo does (all files live in the repo root):

```
cleancut/
├── app.py               # Main entry point — run the program from here
├── lyrics_fetcher.py    # Downloads lyrics via the Genius API
├── aligner.py           # Aligns audio with lyrics word-by-word using WhisperX
├── content_checker.py   # Detects profanity/inappropriate content using Ollama/Gemma
├── cutter.py             # Cuts the audio with a BPM-aware crossfade
├── system_prompt.txt    # Instructions sent to the LLM (you can edit this freely)
├── .gitignore
├── requirements.txt     # ⚠️ Not in the repo yet — see "Setup" below on how to create it
├── .env                 # Where you put your API keys — you need to create this yourself
│
├── music/               # Put the songs you want to process here (created on first run)
│   └── song title - artist.mp3
│
├── sources/             # Lyrics downloaded from Genius are saved here (auto-created)
│   └── song_title-artist_lyrics.txt
│
└── cleancutted/         # Final cleaned audio files end up here (auto-created)
    └── song_title-artist_cleancutted.mp3
```

### What each file does (summary)

| File | What it does |
|---|---|
| `app.py` | The application's entry point. Scans the music folder, lets the user pick a song, and calls all the other modules in sequence to run the whole process end to end. |
| `lyrics_fetcher.py` | Downloads lyrics from the Genius API using the song title and artist, saves them as a text file in `sources/`, and reads them line by line. |
| `aligner.py` | Uses WhisperX to downsample the audio to 16 kHz and produce word-level timestamps (start/end) aligned with the downloaded lyrics. |
| `content_checker.py` | Sends the lyric lines to a locally running Ollama `gemma3:4b` model and flags profane/inappropriate phrases. Reads its instructions from `system_prompt.txt` (with hot-reload support). |
| `cutter.py` | Finds the audio time ranges for the flagged phrases, calculates the song's BPM with `librosa`, snaps the cut points to the nearest beat, and produces the final file by applying a BPM-adjusted crossfade with `pydub`. |
| `system_prompt.txt` | The instruction text that defines what the LLM should consider "inappropriate." The app re-reads this file on every run. |

---

## Requirements (Before You Install)

Before running the app, make sure the following are installed on your system:

| Requirement | What it's for | Note |
|---|---|---|
| **Python 3.9+** | Runtime for the app | Recommended due to type hints (`list[...]`, `tuple[...]`) used in the code |
| **FFmpeg** | Audio processing (used internally by `pydub`) | Must be installed and added to your system `PATH` |
| **Ollama** | Runs the local LLM | After installing, pull the model with `ollama run gemma3:4b` |
| **PyTorch** | A dependency of WhisperX | Installation differs depending on CPU vs CUDA/GPU (see below) |
| **Genius API Key** | Needed to download lyrics | Get a free one at [genius.com/api-clients](https://genius.com/api-clients) |

### Python libraries used

These are the packages imported in the code, and therefore need to be installed:

- `python-dotenv` — for reading the `.env` file
- `whisperx` — for word-level audio–lyrics alignment
- `soundfile`, `numpy` — for audio data processing
- `librosa` — for BPM/beat detection
- `pydub` — for cutting audio and applying crossfades (requires FFmpeg)
- A Genius API client (`lyricsgenius` or `requests`) — for downloading lyrics
- An Ollama client (`ollama` package or `requests`) — for talking to the local LLM

> **Note:** There is no `requirements.txt` in the repo yet. The Setup section below explains how to create one.

---

## Setup

1. Clone the repo:

   ```bash
   git clone https://github.com/baranozc/cleancut.git
   cd cleancut
   ```

2. Since `requirements.txt` doesn't exist in the repo, create one yourself with the following content:

   ```txt
   python-dotenv
   whisperx
   soundfile
   numpy
   librosa
   pydub
   lyricsgenius
   ollama
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** For best WhisperX performance, it's recommended to install PyTorch separately, matched to your CUDA version (see [pytorch.org](https://pytorch.org/get-started/locally/)).

4. Install FFmpeg and add it to your PATH (e.g. [ffmpeg.org](https://ffmpeg.org/download.html) on Windows, `brew install ffmpeg` on macOS, `apt install ffmpeg` on Linux).

5. Install [Ollama](https://ollama.com/download) and pull the model:

   ```bash
   ollama run gemma3:4b
   ```

---

## Usage

1. Create a `.env` file in the project root and add your Genius API key:

   ```env
   GENIUS_API_KEY=your_genius_token_here
   ```

2. Put the song you want to process in the `music/` folder. The filename should follow this format:

   ```
   song title - artist.mp3
   ```

   Supported formats: `.mp3`, `.wav`, `.flac`, `.m4a`, `.ogg`, `.aac`

3. Run the app:

   ```bash
   python app.py
   ```

4. Pick a song from the list by its number — everything else runs automatically.

---

## System Prompt

You can freely edit `system_prompt.txt`. This file contains the instructions the LLM uses to decide what counts as profane/inappropriate content, and is re-read on every run without needing to restart the app (hot-reload).

---

## Processing Steps

```
1. Scan the music/ folder → display the list of songs
2. User picks a song
3. Download lyrics from the Genius API → save as TXT in sources/
4. Align at the word level using WhisperX (16 kHz, CPU)
5. Detect profanity/inappropriate content using Ollama/gemma3:4b
6. Match flagged words/phrases with their timestamps
7. Apply cuts with a BPM-adjusted crossfade
8. Save the final file to cleancutted/
```

---

## Notes

- Cutting is performed on the original (uncompressed) audio file, so sound quality is preserved.
- Crossfade duration is calculated dynamically based on the song's BPM (min 80ms, max 600ms).
- LLM instructions are read from `system_prompt.txt` and can be updated without restarting the app.
- Downloaded lyrics are stored permanently in the `sources/` folder, so they don't need to be re-downloaded.

## Known Gaps

- The repo doesn't yet include `requirements.txt` or `.env.example` — you can create them yourself using the content provided in this README.
- The `music/`, `sources/`, and `cleancutted/` folders aren't part of the repo; the app creates them automatically on first run.

DEVELOPED WITH AI ASSISTANCE.
