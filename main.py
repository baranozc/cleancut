#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Ensure the project root is in Python's search path
root_dir = Path(__file__).parent.resolve()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Import only from core
from core.models import DEFAULT_CUT_MODE, CutMode
from core.pipeline import run_pipeline

SUPPORTED_EXT = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}

def banner():
    print("""
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║   ██████╗██╗     ███████╗ █████╗ ███╗   ██╗ ██████╗██╗   ██╗████████╗  ║
║  ██╔════╝██║     ██╔════╝██╔══██╗████╗  ██║██╔════╝██║   ██║╚══██╔══╝  ║
║  ██║     ██║     █████╗  ███████║██╔██╗ ██║██║     ██║   ██║   ██║     ║
║  ██║     ██║     ██╔══╝  ██╔══██║██║╚██╗██║██║     ██║   ██║   ██║     ║
║  ╚██████╗███████╗███████╗██║  ██║██║ ╚████║╚██████╗╚██████╔╝   ██║     ║
║   ╚═════╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═════╝    ╚═╝     ║
║                                                                        ║
║         🎵  CleanCut — Otomatik Şarkı Temizleme Sistemi 🎵             ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
""")

def on_progress(step: str, message: str) -> None:
    """Progress callback passed to the core pipeline to report progress in console."""
    ts = time.strftime("%H:%M:%S")
    step_label = step.upper().center(15)
    print(f"[{ts}] [{step_label}] ▶  {message}")

def parse_song_artist(filename: str) -> tuple[str, str]:
    """Extracts song name and artist name from the file name if format is 'song - artist'."""
    stem = Path(filename).stem
    if " - " in stem:
        parts = stem.split(" - ", 1)
        return parts[0].strip(), parts[1].strip()
    return stem.strip(), ""

def list_music_files(music_dir: Path) -> list[Path]:
    """Lists supported audio files inside music_dir."""
    if not music_dir.exists():
        return []
    files = [
        f for f in music_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXT
    ]
    return sorted(files, key=lambda f: f.name.lower())

def select_song(files: list[Path]) -> Path:
    """Prompts the user to select a song by number from the list."""
    print("\n┌── Müzik Klasörü ──────────────────────────────────────────┐")
    for i, f in enumerate(files, 1):
        song, artist = parse_song_artist(f.name)
        if artist:
            label = f"{song}  ─  {artist}"
        else:
            label = song
        print(f"│  {i:>3}.  {label}")
    print("└──────────────────────────────────────────────────────────┘")

    while True:
        try:
            choice = int(input("\nNumara seçin: ").strip())
            if 1 <= choice <= len(files):
                return files[choice - 1]
        except ValueError:
            pass
        print("  Geçersiz seçim, lütfen tekrar deneyin.")

def main():
    banner()

    parser = argparse.ArgumentParser(description="CleanCut: Automated Song Profanity & Content Censoring System")
    parser.add_argument("audio_path", nargs="?", default=None, help="Path to the input audio file (optional)")
    parser.add_argument("song_name", nargs="?", default=None, help="Name of the song (optional)")
    parser.add_argument("artist_name", nargs="?", default=None, help="Name of the artist (optional)")
    parser.add_argument("--device", default="cpu", help="Device to use for WhisperX (cpu or cuda)")
    parser.add_argument("--language", default="tr", help="Language code of the lyrics (default: tr)")
    parser.add_argument("--model", default="gemma4:e4b", help="Ollama LLM model to use (default: gemma4:e4b)")
    parser.add_argument("--ollama-url", default="http://localhost:11434/v1/chat/completions", help="Ollama endpoint URL")
    parser.add_argument(
        "--cut-mode", default=DEFAULT_CUT_MODE.value, choices=[m.value for m in CutMode],
        help=" | ".join(f"{m.value}: {m.label} - {m.description}" for m in CutMode)
             + f" (default: {DEFAULT_CUT_MODE.value})",
    )
    args = parser.parse_args()

    # Paths setup
    music_dir = root_dir / "music"
    sources_dir = str(root_dir / "sources")
    cleancutted_dir = root_dir / "cleancutted"

    # Load environmental variables
    env_path = root_dir / ".env"
    if not env_path.exists():
        env_path = root_dir.parent / ".env"
    load_dotenv(dotenv_path=env_path)

    genius_token = os.getenv("GENIUS_API_KEY") or os.getenv("GENIUS_TOKEN")
    if not genius_token:
        print(f"[ERROR] ✖ Hata: .env dosyasında GENIUS_API_KEY veya GENIUS_TOKEN bulunamadı!", file=sys.stderr)
        sys.exit(1)

    # Load system prompt
    prompt_path = root_dir / "system_prompt.txt"
    if not prompt_path.exists():
        print(f"[ERROR] ✖ Hata: system_prompt.txt bulunamadı! Yol: {prompt_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
    except Exception as e:
        print(f"[ERROR] ✖ Hata: system_prompt.txt dosyası okunamadı: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine input parameters (Direct Arguments vs Interactive Mode)
    if args.audio_path:
        audio_file_path = args.audio_path
        song_name = args.song_name
        artist_name = args.artist_name

        if not song_name or not artist_name:
            parsed_song, parsed_artist = parse_song_artist(os.path.basename(audio_file_path))
            if not song_name:
                song_name = parsed_song
            if not artist_name:
                artist_name = parsed_artist
    else:
        # Enters Interactive Mode
        print(f"[{time.strftime('%H:%M:%S')}] ▶  Müzik klasörü taranıyor…")
        music_dir.mkdir(exist_ok=True)
        
        music_files = list_music_files(music_dir)
        if not music_files:
            print(f"[ERROR] ✖ '{music_dir}' klasöründe desteklenen müzik dosyası bulunamadı.\n"
                  f"  Desteklenen formatlar: {', '.join(SUPPORTED_EXT)}\n"
                  f"  Lütfen müzik dosyalarını bu klasöre ekleyip tekrar çalıştırın.", file=sys.stderr)
            sys.exit(1)

        print(f"[{time.strftime('%H:%M:%S')}] ✔  {len(music_files)} şarkı bulundu.")
        
        selected_file = select_song(music_files)
        audio_file_path = str(selected_file)

        parsed_song, parsed_artist = parse_song_artist(selected_file.name)
        song_name = parsed_song
        artist_name = parsed_artist

        if not song_name:
            song_name = input("Şarkı adı bulunamadı, lütfen girin: ").strip()
        if not artist_name:
            artist_name = input("Şarkıcı adı bulunamadı, lütfen girin: ").strip()

    if not os.path.exists(audio_file_path):
        print(f"[ERROR] ✖ Hata: Ses dosyası bulunamadı: {audio_file_path}", file=sys.stderr)
        sys.exit(1)

    cleancutted_dir.mkdir(exist_ok=True)
    safe_name = f"{song_name}-{artist_name}".replace(" ", "_")
    # Çıktı, kalite kaybı olmasın diye kaynakla aynı formatta yazılır
    output_path = str(cleancutted_dir / f"{safe_name}_cleancutted{Path(audio_file_path).suffix}")

    print("\n═" * 74)
    print(f"  • Ses Dosyası  : {audio_file_path}")
    print(f"  • Şarkı Adı    : {song_name}")
    print(f"  • Sanatçı      : {artist_name}")
    print(f"  • Cihaz        : {args.device}")
    print(f"  • Model        : {args.model}")
    print(f"  • Kesim modu   : {CutMode(args.cut_mode).label} ({args.cut_mode})")
    print("═" * 74 + "\n")

    # Run core pipeline
    result = run_pipeline(
        audio_path=audio_file_path,
        song_name=song_name,
        artist_name=artist_name,
        genius_token=genius_token,
        system_prompt=system_prompt,
        sources_dir=sources_dir,
        output_path=output_path,
        on_progress=on_progress,
        language_code=args.language,
        device=args.device,
        model_name=args.model,
        ollama_url=args.ollama_url,
        cut_mode=args.cut_mode,
    )

    print("\n" + "═" * 74)
    if not result.success:
        print(f"[{time.strftime('%H:%M:%S')}] ✖  İŞLEM BAŞARISIZ OLDU!", file=sys.stderr)
        print(f"Hata Mesajı : {result.message}", file=sys.stderr)
        if result.error:
            print(f"Hata Detayı : {result.error}", file=sys.stderr)
        sys.exit(1)

    # Success output display
    print(f"[{time.strftime('%H:%M:%S')}] ✔  İŞLEM BAŞARIYLA TAMAMLANDI!")
    print(f"Mesaj       : {result.message}")
    
    flagged = result.data.get("flagged", [])
    cuts = result.data.get("cuts", [])
    output_file = result.data.get("output_file")

    if flagged:
        print(f"İşaretlenen : {len(flagged)} kelime/kelime grubu")
        for f_word in flagged[:10]:
            print(f"  - {f_word}")
        if len(flagged) > 10:
            print(f"  ...ve {len(flagged) - 10} adet daha.")
    
    if cuts:
        print(f"Uygulanan   : {result.data.get('cuts_applied', 0)} adet beat-aware kesim aralığı")
        print(f"Çıktı Yolu  : {output_file}")
    else:
        print("Uygulanan   : Kesim yapılmadı (Şarkıda uygunsuz içerik bulunamadı veya eşleştirilemedi)")

if __name__ == "__main__":
    main()
