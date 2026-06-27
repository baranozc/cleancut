import os
import re
import gc
import lyricsgenius
from core.models import StepResult

def fetch_and_save_lyrics(song_name: str, artist_name: str, sources_dir: str, genius_token: str) -> StepResult:
    """
    Şarkı sözlerini Genius'tan çekip sources_dir/{song_name}-{artist_name} lyrics.txt
    dosyasına kaydeder. Dosya mevcutsa tekrar indirmez.
    """
    genius = None
    song = None
    try:
        safe_song   = re.sub(r'[\\/*?:"<>|]', '', song_name).strip()
        safe_artist = re.sub(r'[\\/*?:"<>|]', '', artist_name).strip()
        filename    = f"{safe_song}-{safe_artist} lyrics.txt"
        filepath    = os.path.join(sources_dir, filename)

        os.makedirs(sources_dir, exist_ok=True)

        if os.path.exists(filepath):
            load_res = load_lyrics_lines(filepath)
            if load_res.success:
                lines = load_res.data.get("lyrics_lines", [])
                return StepResult(
                    success=True,
                    data={"filepath": filepath, "lyrics_lines": lines, "cached": True},
                    message=f"Mevcut dosya kullanılıyor: {filename}"
                )
            else:
                return load_res

        genius = lyricsgenius.Genius(
            genius_token,
            remove_section_headers=True,
            skip_non_songs=True,
            timeout=15,
        )

        song = genius.search_song(song_name, artist_name)
        if not song:
            return StepResult(
                success=False,
                message="Şarkı Genius'ta bulunamadı."
            )

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

        return StepResult(
            success=True,
            data={"filepath": filepath, "lyrics_lines": lines, "cached": False},
            message=f"{len(lines)} satır kaydedildi → {filename}"
        )
    except Exception as e:
        return StepResult(
            success=False,
            message="Şarkı sözleri Genius'tan çekilirken hata oluştu.",
            error=str(e)
        )
    finally:
        if genius is not None:
            del genius
        if song is not None:
            del song
        gc.collect()

def load_lyrics_lines(filepath: str) -> StepResult:
    """Kaydedilmiş lyrics dosyasından satırları okur."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        return StepResult(
            success=True,
            data={"lyrics_lines": lines},
            message=f"Sözler başarıyla okundu: {len(lines)} satır."
        )
    except Exception as e:
        return StepResult(
            success=False,
            message=f"Lyrics dosyası okunurken hata oluştu: {filepath}",
            error=str(e)
        )
    finally:
        gc.collect()
