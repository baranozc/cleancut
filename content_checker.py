"""
content_checker.py
------------------
Ollama üzerinden yerel Gemma modeline şarkı sözlerini gönderip
küfür / cinsel içerik tespiti yapar.

Endpoint: OpenAI-uyumlu /v1/chat/completions  (405 sorunu olmaz)
Model   : gemma4:e4b
"""

import json
import os
import requests


# OpenAI-uyumlu endpoint — /api/chat yerine bu kullanılır (405 hatası vermez)
OLLAMA_BASE  = "http://localhost:11434"
OLLAMA_URL   = f"{OLLAMA_BASE}/v1/chat/completions"
OLLAMA_MODEL = "gemma4:e4b"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_FILE = os.path.join(_SCRIPT_DIR, "system_prompt.txt")


def _load_system_prompt() -> str:
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()


def check_profanity(lyrics_lines: list[str]) -> list[str]:
    """
    Verilen lyrics satırlarını LLM'e gönderir ve işaretlenmiş
    kelime/kelime-grubu listesini döner.

    Returns:
        flagged: ["kelime1", "kelime grubu", ...]
    """
    system_prompt = _load_system_prompt()
    user_content  = "\n".join(lyrics_lines)

    print(f"  [llm] Ollama'ya gönderiliyor ({OLLAMA_MODEL})…")

    # JSON key adını pekiştirmek için kullanıcı mesajına not ekle
    user_message = (
        user_content
        + '\n\n[IMPORTANT: You MUST use exactly the key "flagged" in your response. '
        'Do NOT use any other key name (flagged_list, flaggedWords, etc.).]'
    )

    # OpenAI-uyumlu format
    payload = {
        "model":  OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        # JSON modu — modelin yalnızca JSON döndürmesini zorlar
        "response_format": {"type": "json_object"},
    }

    headers = {
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, headers=headers, timeout=180)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"  [llm] HATA – Ollama'ya bağlanılamadı. 'ollama serve' çalışıyor mu?")
        return []
    except requests.exceptions.HTTPError as e:
        print(f"  [llm] HATA – HTTP {resp.status_code}: {resp.text[:300]}")
        return []
    except requests.RequestException as e:
        print(f"  [llm] HATA – {e}")
        return []

    # OpenAI-uyumlu yanıt: choices[0].message.content
    raw = resp.json()
    try:
        content = raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        print(f"  [llm] UYARI – Beklenmeyen yanıt yapısı:\n{json.dumps(raw, indent=2)[:400]}")
        return []

    # JSON ayrıştır
    try:
        # Modelin bazen ```json ... ``` bloğuna sardığı olur — temizle
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

        print(f"  [llm] {len(flagged)} öğe işaretlendi.")
        return flagged
    except json.JSONDecodeError:
        print(f"  [llm] UYARI – JSON ayrıştırılamadı:\n{content[:400]}")
        return []
