"""
lyrics_fetcher.py
-----------------
Genius API üzerinden şarkı sözlerini indirip 'sources' klasörüne kaydeder.
"""

import re
import os
import lyricsgenius


def fetch_and_save_lyrics(song_name: str, artist_name: str, sources_dir: str, genius_token: str) -> str | None:
    """
    Şarkı sözlerini Genius'tan çekip sources_dir/{song_name}-{artist_name} lyrics.txt
    dosyasına kaydeder. Dosya mevcutsa tekrar indirmez.

    Returns: lyrics dosyasının tam yolu, yoksa None.
    """
    safe_song   = re.sub(r'[\\/*?:"<>|]', '', song_name).strip()
    safe_artist = re.sub(r'[\\/*?:"<>|]', '', artist_name).strip()
    filename    = f"{safe_song}-{safe_artist} lyrics.txt"
    filepath    = os.path.join(sources_dir, filename)

    os.makedirs(sources_dir, exist_ok=True)

    if os.path.exists(filepath):
        print(f"  [lyrics] Mevcut dosya kullanılıyor: {filename}")
        return filepath

    print(f"  [lyrics] Aranıyor: {song_name} – {artist_name}")
    genius = lyricsgenius.Genius(
        genius_token,
        remove_section_headers=True,
        skip_non_songs=True,
        timeout=15,
    )

    song = genius.search_song(song_name, artist_name)
    if not song:
        print("  [lyrics] Şarkı Genius'ta bulunamadı.")
        return None

    lyrics = song.lyrics

    # 1. İlk satır "Şarkı Adı Lyrics" formatında gelir — kaldır
    lyrics = re.sub(r'^.*?Lyrics\n', '', lyrics, flags=re.IGNORECASE)
    # 2. Sondaki "Embed" / rakam+Embed kalıntılarını kaldır
    lyrics = re.sub(r'\d*Embed$', '', lyrics.strip())
    lyrics = re.sub(r'Embed$',    '', lyrics.strip())
    # 3. Boş satırları temizle
    lines = [line.strip() for line in lyrics.splitlines() if line.strip()]

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"  [lyrics] {len(lines)} satır kaydedildi → {filename}")
    return filepath


def load_lyrics_lines(filepath: str) -> list[str]:
    """Kaydedilmiş lyrics dosyasından satırları okur."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]
