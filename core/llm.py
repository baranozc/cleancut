import json
import gc
import requests
from core.models import StepResult

def check_profanity(
    lyrics_lines: list[str],
    system_prompt: str,
    ollama_url: str = "http://localhost:11434/v1/chat/completions",
    model_name: str = "gemma4:e4b",
) -> StepResult:
    """
    Verilen lyrics satırlarını LLM'e gönderir ve işaretlenmiş
    kelime/kelime-grubu listesini döner.
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
            "response_format": {"type": "json_object"},
        }

        headers = {
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(ollama_url, json=payload, headers=headers, timeout=180)
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
            clean = content.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
                clean = clean.strip()

            data = json.loads(clean)

            # "flagged" anahtarını birkaç olası adla ara (model tutarsızlığına karşı)
            flagged = None
            for key in ("flagged", "flagged_list", "flaggedWords", "flaggedItems",
                        "words", "items", "results"):
                if key in data and isinstance(data[key], list):
                    flagged = data[key]
                    break
            if flagged is None:
                # Listeyi doğrudan dönen modeller için (root level list)
                for val in data.values():
                    if isinstance(val, list):
                        flagged = val
                        break
            flagged = flagged or []

            return StepResult(
                success=True,
                data={"flagged": flagged},
                message=f"{len(flagged)} öğe işaretlendi."
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
