"""
Ses dosyalarını kalite kaybı olmadan okuma/yazma (ffmpeg).

Ses 32-bit float olarak işlenir ve kaynakla aynı formatta, aynı örnekleme hızı,
kanal sayısı ve bit derinliğiyle geri yazılır:
  - Kayıpsız formatlar (FLAC, WAV, ALAC) kayıpsız kalır; kesimlerin dışındaki
    bölümler kaynakla bit bit aynıdır.
  - Kayıplı formatlar (MP3, AAC, Vorbis, Opus) codec'in en yüksek kalitesiyle kodlanır.
Etiketler, kapak resmi ve stream etiketleri kaynaktan kopyalanır. Kesimden sonra
geçersiz kalacak süre etiketi (TLEN) ve bölüm (chapter) zamanları kopyalanmaz.
"""
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from pydub import AudioSegment
from pydub.utils import get_prober_name

# Kayıplı codec'ler için en yüksek kalite ayarları
_LOSSY_ARGS = {
    "mp3":    ["-c:a", "libmp3lame", "-b:a", "320k"],
    "aac":    ["-c:a", "aac", "-b:a", "320k"],
    "vorbis": ["-c:a", "libvorbis", "-q:a", "10"],
    "opus":   ["-c:a", "libopus", "-b:a", "510k"],
}
# Kapak resmini (attached picture) taşıyabilen kapsayıcılar
_COVER_EXTS = {".mp3", ".flac", ".m4a", ".mp4"}


@dataclass
class AudioInfo:
    sample_rate: int
    channels: int
    codec: str
    sample_fmt: str
    bits: int | None   # kayıpsız formatlarda örnek başına bit
    has_cover: bool


def _run(args: list[str], stdin: bytes | None = None) -> bytes:
    proc = subprocess.run(args, input=stdin, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace").strip()[-600:])
    return proc.stdout


def probe(path: str) -> AudioInfo:
    data = json.loads(_run([get_prober_name(), "-v", "error", "-print_format", "json",
                            "-show_streams", path]))
    audio = next(s for s in data["streams"] if s["codec_type"] == "audio")
    has_cover = any(s["codec_type"] == "video" and s.get("disposition", {}).get("attached_pic")
                    for s in data["streams"])
    bits = int(audio.get("bits_per_raw_sample") or 0) or int(audio.get("bits_per_sample") or 0) or None
    return AudioInfo(
        sample_rate=int(audio["sample_rate"]),
        channels=int(audio["channels"]),
        codec=audio["codec_name"],
        sample_fmt=audio.get("sample_fmt", ""),
        bits=bits,
        has_cover=has_cover,
    )


def decode(path: str, info: AudioInfo) -> np.ndarray:
    """(örnek, kanal) biçiminde float32 dizi döner; örnekleme hızı ve kanallar değişmez."""
    raw = _run([AudioSegment.converter, "-v", "error", "-i", path, "-map", "0:a:0",
                "-f", "f32le", "-acodec", "pcm_f32le", "-"])
    return np.frombuffer(raw, dtype="<f4").reshape(-1, info.channels)


def _codec_args(info: AudioInfo) -> list[str]:
    if info.codec in _LOSSY_ARGS:
        return _LOSSY_ARGS[info.codec]
    if info.codec.startswith("pcm_"):
        return ["-c:a", info.codec]
    if info.codec in ("flac", "alac"):
        planar = "p" if info.codec == "alac" else ""
        if (info.bits or 16) <= 16:
            return ["-c:a", info.codec, "-sample_fmt", f"s16{planar}"]
        return ["-c:a", info.codec, "-sample_fmt", f"s32{planar}", "-bits_per_raw_sample", str(info.bits)]
    raise ValueError(f"Desteklenmeyen ses codec'i: {info.codec}")


def encode_like(samples: np.ndarray, info: AudioInfo, source_path: str, output_path: str) -> None:
    """samples'ı kaynakla aynı codec/kalitede output_path'e yazar; metadata'yı kaynaktan kopyalar."""
    args = [AudioSegment.converter, "-v", "error", "-y",
            "-f", "f32le", "-ar", str(info.sample_rate), "-ac", str(info.channels), "-i", "pipe:0",
            "-i", source_path,
            "-map", "0:a"]
    ext = Path(output_path).suffix.lower()
    if info.has_cover and ext in _COVER_EXTS:
        args += ["-map", "1:v", "-c:v", "copy", "-disposition:v", "attached_pic"]
    args += ["-map_metadata", "1", "-map_metadata:s:a", "1:s:a", "-map_chapters", "-1",
             "-metadata", "TLEN="]
    args += _codec_args(info)
    if ext == ".mp3":
        args += ["-id3v2_version", "3"]
    args.append(output_path)
    _run(args, stdin=np.ascontiguousarray(samples, dtype="<f4").tobytes())
