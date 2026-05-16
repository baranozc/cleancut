#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║   ██████╗██╗     ███████╗ █████╗ ███╗   ██╗ ██████╗██╗   ██╗████████╗  ║
║  ██╔════╝██║     ██╔════╝██╔══██╗████╗  ██║██╔════╝██║   ██║╚══██╔══╝  ║
║  ██║     ██║     █████╗  ███████║██╔██╗ ██║██║     ██║   ██║   ██║     ║
║  ██║     ██║     ██╔══╝  ██╔══██║██║╚██╗██║██║     ██║   ██║   ██║     ║
║  ╚██████╗███████╗███████╗██║  ██║██║ ╚████║╚██████╗╚██████╔╝   ██║     ║
║   ╚═════╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═════╝    ╚═╝     ║
║                                                                        ║
║          🎵  Otomatik Şarkı Temizleme Sistemi  🎵                     ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝

app.py — Ana konsol uygulaması
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# ── Yerel modüller ────────────────────────────────────────────────────────────
from lyrics_fetcher  import fetch_and_save_lyrics, load_lyrics_lines
from aligner         import align_lyrics, build_line_timestamps
from content_checker import check_profanity
from cutter          import find_flagged_timestamps, apply_cuts

# ── Klasör yapısı ────────────────────────────────────────────────────────────
APP_DIR      = Path(__file__).parent          # cleancut_app/
MUSIC_DIR    = APP_DIR / "music"              # müzik dosyaları buraya
SOURCES_DIR  = APP_DIR / "sources"           # lyrics txt dosyaları
CLEANCUTTED  = APP_DIR / "cleancutted"       # temizlenmiş çıktılar

SUPPORTED_EXT = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}

# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────

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
║         🎵  Hoş Geldiniz! CleanCut v1.0  🎵                           ║
║         Küfür & İçerik Temizleme Otomasyonu                            ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
""")


def step(msg: str):
    """Adım mesajını renkli ve tarih damgalı yazdırır."""
    ts = time.strftime("%H:%M:%S")
    print(f"\n[{ts}] ▶  {msg}")


def success(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] ✔  {msg}")


def warn(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] ⚠  {msg}")


def error_exit(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] ✖  {msg}")
    sys.exit(1)


def parse_song_artist(filename: str) -> tuple[str, str]:
    """
    Dosya adından 'şarkı adı - şarkıcı' formatını çıkartmaya çalışır.
    Bulamazsa (şarkı_adı, '') döner.
    """
    stem = Path(filename).stem  # uzantısız ad
    if " - " in stem:
        parts = stem.split(" - ", 1)
        return parts[0].strip(), parts[1].strip()
    return stem.strip(), ""


def list_music_files() -> list[Path]:
    """MUSIC_DIR içindeki desteklenen ses dosyalarını listeler."""
    if not MUSIC_DIR.exists():
        return []
    files = [
        f for f in MUSIC_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXT
    ]
    return sorted(files, key=lambda f: f.name.lower())


def select_song(files: list[Path]) -> Path:
    """Kullanıcıdan numara ile şarkı seçtirir."""
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


# ── Ana akış ─────────────────────────────────────────────────────────────────

def main():
    banner()

    # .env yükle
    env_path = APP_DIR / ".env"
    if not env_path.exists():
        # Bir üst dizindeki .env'yi de dene
        env_path = APP_DIR.parent / ".env"
    load_dotenv(dotenv_path=env_path)

    genius_token = os.getenv("GENIUS_API_KEY") or os.getenv("GENIUS_TOKEN")
    if not genius_token:
        error_exit(".env dosyasında GENIUS_API_KEY bulunamadı!")

    # ── 1. Müzik listesi ──────────────────────────────────────────────────────
    step("Müzik klasörü taranıyor…")
    MUSIC_DIR.mkdir(exist_ok=True)
    SOURCES_DIR.mkdir(exist_ok=True)
    CLEANCUTTED.mkdir(exist_ok=True)

    music_files = list_music_files()
    if not music_files:
        error_exit(
            f"'{MUSIC_DIR}' klasöründe desteklenen müzik dosyası bulunamadı.\n"
            f"  Desteklenen formatlar: {', '.join(SUPPORTED_EXT)}\n"
            f"  Lütfen müzik dosyalarını bu klasöre ekleyip tekrar çalıştırın."
        )

    success(f"{len(music_files)} şarkı bulundu.")

    # ── 2. Şarkı seçimi ───────────────────────────────────────────────────────
    selected = select_song(music_files)
    song_name, artist_name = parse_song_artist(selected.name)

    if not artist_name:
        artist_name = input("Şarkıcı adı bulunamadı, lütfen girin: ").strip()
    if not song_name:
        song_name = input("Şarkı adı bulunamadı, lütfen girin: ").strip()

    print(f"\n  Seçilen: «{song_name}» – {artist_name}")
    print(f"  Dosya  : {selected}")

    # ── 3. Lirik indirme ─────────────────────────────────────────────────────
    step("Genius'tan şarkı sözleri indiriliyor…")
    lyrics_path = fetch_and_save_lyrics(
        song_name   = song_name,
        artist_name = artist_name,
        sources_dir = str(SOURCES_DIR),
        genius_token= genius_token,
    )
    if not lyrics_path:
        error_exit("Şarkı sözleri indirilemedi. İşlem durduruluyor.")

    lyrics_lines = load_lyrics_lines(lyrics_path)
    success(f"Şarkı sözleri hazır — {len(lyrics_lines)} satır.")

    # ── 4. WhisperX hizalama ─────────────────────────────────────────────────
    step("WhisperX ile timestamp hizalaması yapılıyor…")
    print("  (Bu adım birkaç dakika sürebilir, lütfen bekleyin…)")
    word_segments = align_lyrics(
        audio_path   = str(selected),
        lyrics_lines = lyrics_lines,
        language_code= "en",
        device       = "cpu",
        compute_type = "int8",
    )

    if not word_segments:
        error_exit("Hizalama başarısız — word_segments boş döndü.")

    line_timestamps = build_line_timestamps(lyrics_lines, word_segments)
    success(f"Hizalama tamamlandı — {len(word_segments)} kelime.")

    # Özet göster
    print("\n  ─── İlk 5 satır (önizleme) ───")
    for lt in line_timestamps[:5]:
        print(f"  {lt['start']:.2f}s – {lt['end']:.2f}s  →  {lt['line']}")

    # ── 5. İçerik denetimi (LLM) ─────────────────────────────────────────────
    step("Yerel LLM (Ollama/Gemma) ile içerik denetimi yapılıyor…")
    flagged = check_profanity(lyrics_lines)

    if not flagged:
        success("Hiçbir uygunsuz içerik tespit edilmedi. Şarkı temiz! 🎉")
        print("\n  İşlem tamamlandı — kesim yapılmadı.")
        return

    print(f"\n  ⚑ İşaretlenen öğeler ({len(flagged)}):")
    for item in flagged:
        print(f"    • {item}")

    # ── 6. Timestamp eşleştirme ───────────────────────────────────────────────
    step("İşaretlenen kelimeler için timestamp'ler çıkartılıyor…")
    cuts = find_flagged_timestamps(flagged, word_segments)

    if not cuts:
        warn("Timestamp eşleştirilemedi — kesim yapılamadı.")
        return

    print(f"\n  Kesilecek {len(cuts)} aralık:")
    for start, end in cuts:
        print(f"    {start:.3f}s  →  {end:.3f}s")

    # ── 7. Crossfade & kesim ──────────────────────────────────────────────────
    step("Beat-aware crossfade ile şarkı kesiliyor (orijinal dosya kullanılıyor)…")
    safe_name   = f"{song_name}-{artist_name}".replace(" ", "_")
    output_path = str(CLEANCUTTED / f"{safe_name}_cleancutted.mp3")

    apply_cuts(
        input_file  = str(selected),
        output_file = output_path,
        cuts        = cuts,
    )

    success(f"Final sürüm kaydedildi: {output_path}")
    print(f"""
╔══════════════════════════════════════════════════════╗
║  ✅  İşlem tamamlandı!                                ║
║                                                      ║
║  Şarkı  : {(song_name + ' – ' + artist_name):<42}║
║  Çıktı  : cleancutted/{Path(output_path).name:<33}║
║  Kesim  : {len(cuts)} aralık temizlendi                          ║
╚══════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
