import json
import gc
import requests
from core.models import StepResult
from core.text import find_sequence, normalize_words

# Aynı şarkı için her seferinde aynı sonucu almak için örnekleme kapalı
# (modelin varsayılan sıcaklığı 1.0, bu da her çalıştırmada farklı işaretlemeye yol açıyordu).
LLM_TEMPERATURE = 0.0
LLM_SEED = 42

# Model cevap vermeden önce sözleri satır satır düşünerek tarıyor; uzun şarkılarda bu
# birkaç dakika sürebilir.
LLM_TIMEOUT_SEC = 600


def _extract_json(content: str):
    """Model cevabından JSON'u çıkarır; ```json bloklarını ve etrafındaki metni tolere eder."""
    clean = content.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
        clean = clean.strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        start, end = clean.find("{"), clean.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(clean[start:end + 1])


def _verify_flagged(flagged: list, lyrics_lines: list[str]) -> tuple[list[str], list[str]]:
    """
    LLM çıktısını temizler ve sözlerle doğrular. Metin olmayan, boş ve tekrarlanan
    öğeler atılır; sözlerde birebir (noktalama/harf büyüklüğü hariç) geçmeyen öğeler
    — model uydurduğunda veya kelimeyi değiştirdiğinde — ayrı listede döner.
    """
    lyric_words = normalize_words(" ".join(lyrics_lines))
    kept: list[str] = []
    rejected: list[str] = []
    seen: set[str] = set()

    for item in flagged:
        if isinstance(item, dict):
            # Bazı modeller [{"text": "..."}] biçiminde döner; ilk metin alanını kullan.
            item = next((v for v in item.values() if isinstance(v, str)), None)
        if not isinstance(item, str):
            continue
        words = normalize_words(item)
        key = " ".join(words)
        if not words or key == "none" or key in seen:
            continue
        seen.add(key)
        if find_sequence(lyric_words, words):
            kept.append(item.strip())
        else:
            rejected.append(item.strip())
    return kept, rejected


def check_profanity(
    lyrics_lines: list[str],
    system_prompt: str,
    ollama_url: str = "http://localhost:11434/v1/chat/completions",
    model_name: str = "gemma4:e4b",
) -> StepResult:
    """
    Verilen lyrics satırlarını LLM'e gönderir ve işaretlenmiş
    kelime/kelime-grubu listesini döner. Sözlerde geçmeyen öğeler
    "flagged" yerine "rejected" listesine konur.
    """
    try:
        user_content  = "\n".join(lyrics_lines)

        # JSON key adını pekiştirmek için kullanıcı mesajına not ekle
        user_message = (
            user_content
            + '\n\n[IMPORTANT: You MUST use exactly the key "flagged" in your response. '
            'Do NOT use any other key name (flagged_list, flaggedWords, etc.).]'
        )

        payload = {
            "model":  model_name,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            # response_format=json_object bilerek kullanılmıyor: Ollama'da açıkken model
            # düşünme aşamasında bulduğu küfürlerin bir kısmını ("fuck", "shit",
            # "orospuları") son cevaba yazmıyordu. JSON'u prompt istiyor, _extract_json ayıklıyor.
            "temperature": LLM_TEMPERATURE,
            "seed": LLM_SEED,
        }

        headers = {
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(ollama_url, json=payload, headers=headers, timeout=LLM_TIMEOUT_SEC)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            return StepResult(
                success=False,
                message="Ollama'ya bağlanılamadı. 'ollama server' çalışıyor mu?",
                error=str(e)
            )
        except requests.exceptions.HTTPError as e:
            return StepResult(
                success=False,
                message=f"Ollama HTTP Hatası: {resp.status_code}",
                error=resp.text[:300]
            )
        except requests.RequestException as e:
            return StepResult(
                success=False,
                message="Ollama isteği sırasında beklenmeyen bir hata oluştu.",
                error=str(e)
            )

        raw = resp.json()
        try:
            content = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            return StepResult(
                success=False,
                message="Ollama yanıt yapısı geçersiz/beklenmeyen bir formatta.",
                error=str(e)
            )

        try:
            data = _extract_json(content)

            # "flagged" anahtarını birkaç olası adla ara (model tutarsızlığına karşı)
            flagged = None
            if isinstance(data, list):
                # Listeyi doğrudan dönen modeller için (root level list)
                flagged = data
            else:
                for key in ("flagged", "flagged_list", "flaggedWords", "flaggedItems",
                            "words", "items", "results"):
                    if key in data and isinstance(data[key], list):
                        flagged = data[key]
                        break
                if flagged is None:
                    for val in data.values():
                        if isinstance(val, list):
                            flagged = val
                            break
            flagged, rejected = _verify_flagged(flagged or [], lyrics_lines)

            message = f"{len(flagged)} öğe işaretlendi."
            if rejected:
                message += f" Sözlerde bulunamadığı için atılanlar: {rejected}"
            return StepResult(
                success=True,
                data={"flagged": flagged, "rejected": rejected},
                message=message
            )
        except json.JSONDecodeError as e:
            return StepResult(
                success=False,
                message="Ollama'dan dönen yanıt JSON olarak ayrıştırılamadı.",
                error=str(e)
            )
    except Exception as e:
        return StepResult(
            success=False,
            message="LLM denetimi sırasında bilinmeyen bir hata oluştu.",
            error=str(e)
        )
    finally:
        gc.collect()
