# CleanCut App

A console application that automatically detects and removes sections containing profanity and explicit content from songs.

## Folder Structure

```
cleancut_app/
├── app.py               # Main application — run from here
├── lyrics_fetcher.py    # Module to fetch lyrics via Genius API
├── aligner.py           # WhisperX timestamp alignment module
├── content_checker.py   # Ollama/Gemma LLM content moderation module
├── cutter.py            # Beat-aware crossfade audio cutting module
├── system_prompt.txt    # LLM system prompt (editable)
├── .env                 # API keys (you need to create this)
├── .env.example         # .env template
├── requirements.txt     # Python dependencies
│
├── music/               # Place the music files to be processed here
│   └── song name - artist.mp3
│
├── sources/             # Lyrics downloaded from Genius are saved here
│   └── song name-artist lyrics.txt
│
└── cleancutted/         # Final cleaned music files are output here
    └── song_name-artist_cleancutted.mp3
```

## Prerequisites

Before running the application, make sure you have the following installed on your system:

- **Python 3.8+**
- **FFmpeg**: Essential for audio processing and manipulation (used by `pydub`). You must install it and add it to your system's PATH.
- **Ollama**: Required for local LLM execution. You need to install Ollama and pull the `gemma3:4b` model (e.g., `ollama run gemma3:4b`).
- **PyTorch**: Required for WhisperX. Installation method may vary depending on your hardware (CPU or CUDA/GPU support).
- **Genius API Key**: Needed to fetch song lyrics.

## Installation

1. Navigate to the app directory:
```bash
cd cleancut_app
```

2. Install the required Python dependencies:
```bash
pip install -r requirements.txt
```

> **Note:** For optimal performance with **WhisperX**, it is recommended to install the PyTorch version that matches your CUDA version.

## Usage

1. Copy `.env.example` to `.env` and insert your Genius API token:
   ```env
   GENIUS_API_KEY=your_genius_token_here
   ```

2. Place the music file you want to process into the `music/` folder. The file name **must** follow the format: **"song name - artist.mp3"**.

3. Run the application:
   ```bash
   python app.py
   ```

4. Select a song from the list by its number — the rest is fully automatic.

## System Prompt

You can edit the `system_prompt.txt` file as you see fit. This file contains the instructions sent to the LLM to govern how it detects profanity and explicit content.

## Processing Workflow

```
1. Scans the music/ folder → displays the song list
2. User selects a song by number
3. Downloads lyrics from Genius API → saves as TXT in the sources/ folder
4. Performs word-level timestamp alignment via WhisperX (16 kHz, CPU)
5. Detects profanity & explicit content using Ollama/gemma3:4b
6. Matches the flagged words/phrases with timestamps
7. Applies cuts using beat-aware crossfade (crossfade length is auto-adjusted by BPM)
8. Saves the final processed file to the cleancutted/ folder
```

## Notes

- The original audio file is used for the final extraction → audio quality is preserved.
- The crossfade duration is dynamically calculated based on the song's BPM.
- The LLM dynamically reads its system prompt from `system_prompt.txt` — supports hot-reloading without restarting the app.
- Downloaded lyrics files are persistently stored in the `sources/` folder.
